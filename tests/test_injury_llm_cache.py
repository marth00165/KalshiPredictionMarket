from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.analysis.injury_llm_cache import (
    InjuryLLMCacheStore,
    InjuryLLMRefreshService,
    build_matchup_injury_snapshot,
    fingerprint_injury_snapshot,
    normalize_injury_status,
    should_refresh_llm,
)


def test_fingerprint_is_stable_across_ordering() -> None:
    payload_a = {
        "teams": [
            {
                "reference": "LAL",
                "players": [
                    {
                        "id": "p2",
                        "full_name": "Second Player",
                        "injuries": [{"status": "Questionable", "desc": "ankle", "comment": "day-to-day"}],
                    },
                    {
                        "id": "p1",
                        "full_name": "First Player",
                        "injuries": [{"status": "Out", "desc": "knee", "comment": ""}],
                    },
                ],
            },
            {
                "reference": "BOS",
                "players": [
                    {
                        "id": "p3",
                        "full_name": "Third Player",
                        "injuries": [{"status": "Probable", "desc": "illness", "comment": ""}],
                    }
                ],
            },
        ]
    }
    payload_b = {
        "teams": [
            payload_a["teams"][1],
            {
                "reference": "LAL",
                "players": [
                    payload_a["teams"][0]["players"][1],
                    payload_a["teams"][0]["players"][0],
                ],
            },
        ]
    }

    snap_a = build_matchup_injury_snapshot(payload_a, home_team="BOS", away_team="LAL")
    snap_b = build_matchup_injury_snapshot(payload_b, home_team="BOS", away_team="LAL")

    assert fingerprint_injury_snapshot(snap_a) == fingerprint_injury_snapshot(snap_b)


def test_status_normalization_mapping() -> None:
    assert normalize_injury_status("Out") == "OUT"
    assert normalize_injury_status("Doubtful - ankle") == "DOUBTFUL"
    assert normalize_injury_status("Questionable") == "QUESTIONABLE"
    assert normalize_injury_status("Probable") == "PROBABLE"
    assert normalize_injury_status("Available") == "AVAILABLE"
    assert normalize_injury_status("") == "UNKNOWN"


def test_team_profile_status_mapping_codes() -> None:
    status_fn = InjuryLLMRefreshService._status_from_profile_player
    assert status_fn({"status": "SUS"}) == "OUT"
    assert status_fn({"status": "DTD"}) == "QUESTIONABLE"
    assert status_fn({"status": "PROB"}) == "PROBABLE"
    assert status_fn({"status": "DOUT"}) == "DOUBTFUL"
    assert status_fn({"status": "suspended by league"}) == "OUT"


def test_team_profile_injury_rows_override_available_status() -> None:
    status_fn = InjuryLLMRefreshService._status_from_profile_player
    player_row = {
        "status": "ACTIVE",
        "injuries": [
            {"status": "Questionable", "desc": "ankle"},
            {"status": "Out", "desc": "knee"},
        ],
    }
    assert status_fn(player_row) == "OUT"


def test_numeric_team_reference_maps_to_alias() -> None:
    payload = {
        "teams": [
            {
                "reference": "1610612763",  # MEM
                "players": [
                    {
                        "id": "p_mem",
                        "full_name": "Mem Player",
                        "injuries": [{"status": "Out", "desc": "knee", "comment": ""}],
                    }
                ],
            },
            {
                "reference": "1610612758",  # SAC
                "players": [
                    {
                        "id": "p_sac",
                        "full_name": "Sac Player",
                        "injuries": [{"status": "Questionable", "desc": "ankle", "comment": ""}],
                    }
                ],
            },
        ]
    }

    snapshot = build_matchup_injury_snapshot(payload, home_team="MEM", away_team="SAC")
    team_players = {row["team"]: row["players"] for row in snapshot["teams"]}
    assert len(team_players.get("MEM", [])) == 1
    assert len(team_players.get("SAC", [])) == 1


def test_should_refresh_llm_decisions() -> None:
    now = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)
    fp = "abc123"

    miss = should_refresh_llm(cache_entry=None, current_injury_fingerprint=fp, now=now)
    assert miss.should_refresh is True
    assert miss.reason == "cache_miss"

    fresh_entry = {
        "injury_fingerprint": fp,
        "generated_at": (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        "last_market_price_seen": 0.50,
    }
    reuse = should_refresh_llm(
        cache_entry=fresh_entry,
        current_injury_fingerprint=fp,
        now=now,
        current_market_price=0.505,
        max_age_seconds=1800,
        price_move_threshold_pct=0.03,
    )
    assert reuse.should_refresh is False
    assert reuse.reason == "cache_hit_reuse"

    changed = should_refresh_llm(
        cache_entry=fresh_entry,
        current_injury_fingerprint="different",
        now=now,
    )
    assert changed.should_refresh is True
    assert changed.reason == "fingerprint_changed"

    old_entry = dict(fresh_entry)
    old_entry["generated_at"] = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    expired = should_refresh_llm(
        cache_entry=old_entry,
        current_injury_fingerprint=fp,
        now=now,
        max_age_seconds=1800,
    )
    assert expired.should_refresh is True
    assert expired.reason == "ttl_expired"

    near_tipoff = should_refresh_llm(
        cache_entry=fresh_entry,
        current_injury_fingerprint=fp,
        now=now,
        tipoff_time=now + timedelta(minutes=20),
        near_tipoff_minutes=45,
        near_tipoff_stale_seconds=60,
    )
    assert near_tipoff.should_refresh is True
    assert near_tipoff.reason == "near_tipoff_stale"

    price_move = should_refresh_llm(
        cache_entry=fresh_entry,
        current_injury_fingerprint=fp,
        now=now,
        current_market_price=0.60,
        price_move_threshold_pct=0.03,
    )
    assert price_move.should_refresh is True
    assert price_move.reason == "large_price_move"


def test_cache_roundtrip(tmp_path) -> None:
    cache_file = tmp_path / "injury_cache.json"
    store = InjuryLLMCacheStore(str(cache_file))
    payload = {
        "matchup_key": "KXNBAGAME-26FEB23LALBOS-LAL",
        "injury_fingerprint": "fingerprint1",
        "llm_delta": -12.0,
        "generated_at": "2026-02-23T00:00:00Z",
    }
    store.upsert(payload["matchup_key"], payload)

    store2 = InjuryLLMCacheStore(str(cache_file))
    loaded = store2.get(payload["matchup_key"])
    assert loaded is not None
    assert loaded["injury_fingerprint"] == "fingerprint1"
    assert float(loaded["llm_delta"]) == -12.0
