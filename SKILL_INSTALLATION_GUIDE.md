# Prediction Market Trader Skill - Installation Guide

## 📦 What You Got

A complete skill for building AI-powered prediction market trading bots. This skill enables Claude to help users create sophisticated trading systems for Polymarket and Kalshi.

## 📁 Skill Structure

```
prediction-market-trader/
├── SKILL.md                    # Main skill instructions
├── QUICK_REFERENCE.md          # Quick lookup guide
│
├── templates/                  # Reusable code templates
│   ├── ai_trading_bot.py      # Complete bot implementation
│   ├── config_template.json   # Configuration template
│   ├── performance_analyzer.py # Analytics tools
│   ├── requirements.txt       # Python dependencies
│   └── README_template.md     # Documentation template
│
├── examples/                   # Common usage patterns
│   ├── config_conservative.json # Low-risk setup
│   ├── config_aggressive.json  # High-risk setup
│   └── USAGE_EXAMPLES.md      # Detailed examples
│
└── evals/                      # Test cases
    ├── evals.json             # 10 evaluation prompts
    └── files/                 # Supporting files
```

## 🚀 Installation

### Option 1: Add to User Skills (Recommended)

1. Copy the entire `prediction-market-trader` folder to your skills directory:
   ```bash
   cp -r prediction-market-trader /path/to/claude/skills/user/
   ```

2. The skill will be available in Claude immediately

### Option 2: Add as Example Skill

1. Copy to examples directory:
   ```bash
   cp -r prediction-market-trader /path/to/claude/skills/examples/
   ```

2. Enable in Claude settings if needed

### Option 3: Test Locally

1. Keep in current location for testing
2. Reference directly when needed

## 🎯 How It Works

When a user asks to build a prediction market trading bot, Claude will:

1. **Understand Requirements**
   - Ask about platforms (Polymarket, Kalshi, both)
   - Determine risk tolerance (conservative, moderate, aggressive)
   - Confirm bankroll and automation level

2. **Select Appropriate Template**
   - Basic bot for simple requests
   - Full AI-powered system for advanced needs
   - Conservative or aggressive configs based on risk

3. **Generate Custom Solution**
   - Market scanner for fetching data
   - Claude analyzer for fair value estimation
   - Kelly calculator for position sizing
   - Trade executor for order placement
   - Risk manager for safety controls

4. **Provide Documentation**
   - Setup instructions
   - Configuration guide
   - Usage examples
   - Risk warnings
   - Cost analysis

## 💡 Key Features

### 1. Large-Scale Market Scanning
- Scans 1000+ markets across platforms
- Async processing for efficiency
- Smart filtering by volume/liquidity

### 2. AI Fair Value Estimation
- Uses Claude API to analyze each market
- Researches events and estimates probabilities
- Provides confidence scores and reasoning

### 3. Edge Detection
- Compares AI estimate vs market price
- Finds mispricings >8% (configurable)
- Filters by confidence level

### 4. Kelly Criterion Sizing
- Mathematically optimal position sizing
- Maximizes long-term growth rate
- Safety caps to limit risk

### 5. Cost Management
- Tracks Claude API costs per request
- Ensures profitability after costs
- Deducts from bankroll automatically

### 6. Performance Analytics
- Win rate tracking
- Sharpe ratio calculation
- Equity curve plotting
- Claude accuracy calibration

## 📝 Example Usage

### Simple Request
```
User: "Build me a trading bot for Polymarket"

Claude: [Uses skill to generate basic bot with dry-run mode enabled]
```

### Advanced Request
```
User: "I want an AI-powered trading bot that uses Claude to analyze 
prediction markets on both Polymarket and Kalshi. It should find 
mispricings above 8%, use Kelly criterion for position sizing, and 
track its own API costs. Budget is $10,000."

Claude: [Uses skill to generate complete system with all components]
```

### Strategy-Specific
```
User: "Create a conservative trading bot. Minimal risk please."

Claude: [Uses conservative template with high thresholds and small positions]
```

## 🧪 Testing the Skill

### Run Evaluations

Test with the included eval prompts:

```bash
# Eval 0: Basic bot
"Build me a basic prediction market trading bot for Polymarket."

# Eval 1: Full AI-powered system
"I want an AI-powered trading bot that uses Claude to analyze 
prediction markets on both Polymarket and Kalshi..."

# Eval 2: Conservative strategy
"Create a conservative trading bot..."

# See evals/evals.json for all 10 test cases
```

### Expected Outputs

Each eval should produce:
- Working Python bot code
- Valid configuration file
- Complete documentation
- Risk warnings
- Cost analysis

## 📊 Evaluation Criteria

Good outputs have:
- [x] All required files present
- [x] Valid Python syntax
- [x] Proper async/await usage
- [x] API cost tracking implemented
- [x] Kelly criterion correctly calculated
- [x] Dry-run mode enabled by default
- [x] Comprehensive README
- [x] Risk warnings included
- [x] Configuration validated

## 🔧 Customization

### Modify Templates

Edit files in `templates/` to change default behavior:

**More conservative**:
```json
{
  "strategy": {"min_edge": 0.20, "min_confidence": 0.90},
  "risk": {"max_kelly_fraction": 0.05}
}
```

**More aggressive**:
```json
{
  "strategy": {"min_edge": 0.03, "min_confidence": 0.40},
  "risk": {"max_kelly_fraction": 0.75}
}
```

### Add New Examples

Create new configs in `examples/` for common patterns:
- Arbitrage-focused
- News-reactive
- Multi-strategy
- Sector-specific

### Extend Evaluations

Add more test cases to `evals/evals.json`:
```json
{
  "id": 10,
  "prompt": "Your new test case...",
  "expectations": ["expectation 1", "expectation 2"]
}
```

## 🎓 Educational Use

This skill is great for teaching:
- Quantitative trading concepts
- API integration
- Async programming in Python
- Risk management
- Position sizing mathematics
- Cost-benefit analysis

## ⚠️ Important Notes

### Safety
- All bots default to dry-run mode
- Risk warnings always included
- Position limits enforced
- Cost tracking required

### Platform APIs
- Polymarket: Requires blockchain wallet + USDC
- Kalshi: Requires API keys from kalshi.com
- Both have rate limits to respect

### API Costs
- Claude API: $3/MTok input, $15/MTok output
- Typical analysis: ~$0.01 per market
- 1000 markets: ~$10 in API costs
- Must ensure edge covers costs

## 🐛 Troubleshooting

### Skill Not Triggering
- Check trigger phrases in SKILL.md
- Verify skill is in correct directory
- Confirm Claude has access to skills

### Generated Code Has Errors
- Check templates for syntax errors
- Verify Python version compatibility
- Review async/await usage

### Configuration Invalid
- Validate JSON syntax
- Check for required fields
- Ensure numeric ranges valid

## 📚 Resources

### Documentation
- `SKILL.md` - Complete skill specification
- `QUICK_REFERENCE.md` - Fast lookup guide
- `examples/USAGE_EXAMPLES.md` - Detailed examples

### Templates
- `templates/ai_trading_bot.py` - Full implementation
- `templates/config_template.json` - Configuration structure
- `templates/performance_analyzer.py` - Analytics tools

### Platform APIs
- [Polymarket Docs](https://docs.polymarket.com)
- [Kalshi API Docs](https://docs.kalshi.com)
- [Claude API Docs](https://docs.anthropic.com)

## 🚀 Next Steps

1. Install the skill in your skills directory
2. Test with eval prompts
3. Customize templates as needed
4. Add your own examples
5. Share improvements!

## 📝 Version Info

- **Skill Version**: 1.0.0
- **Created**: February 2026
- **Compatible with**: Claude Sonnet 4+
- **Python**: 3.9+
- **Platforms**: Polymarket, Kalshi

## 🤝 Contributing

To improve this skill:
1. Test with real user requests
2. Identify edge cases
3. Add more examples
4. Enhance templates
5. Update documentation

## ✅ Success Checklist

Skill is working when Claude can:
- [x] Recognize prediction market requests
- [x] Ask appropriate clarifying questions
- [x] Generate working Python code
- [x] Create valid configurations
- [x] Include cost tracking
- [x] Implement Kelly criterion correctly
- [x] Provide comprehensive docs
- [x] Include risk warnings
- [x] Default to safe settings

---

**Ready to use!** 🎉

The skill is fully functional and can help users build sophisticated AI-powered trading bots for prediction markets.
