"""Error reporting and logging utilities"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from .errors import (
    ValidationError,
    StrategyError,
    PositionError,
    ExecutionError,
    APIDataError,
)

logger = logging.getLogger(__name__)


class ErrorReport:
    """Structured error report for a trading operation"""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.timestamp = datetime.now()
        self.errors: List[Dict[str, Any]] = []
    
    def add_error(self, error: Exception, context: str = "") -> None:
        """Add an error to the report"""
        error_dict = {
            'timestamp': datetime.now(),
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'error_object': error,
        }
        self.errors.append(error_dict)
    
    def has_errors(self) -> bool:
        """Check if any errors recorded"""
        return len(self.errors) > 0
    
    def error_count(self) -> int:
        """Get count of errors"""
        return len(self.errors)
    
    def get_error_types(self) -> List[str]:
        """Get list of unique error types"""
        return list(set(e['type'] for e in self.errors))
    
    def get_errors_by_type(self, error_type: str) -> List[Dict]:
        """Get all errors of a specific type"""
        return [e for e in self.errors if e['type'] == error_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'operation': self.operation,
            'timestamp': self.timestamp.isoformat(),
            'error_count': self.error_count(),
            'error_types': self.get_error_types(),
            'errors': [
                {
                    'type': e['type'],
                    'message': e['message'],
                    'context': e['context'],
                    'timestamp': e['timestamp'].isoformat(),
                }
                for e in self.errors
            ]
        }
    
    def log_summary(self) -> None:
        """Log a summary of the report"""
        if not self.has_errors():
            logger.info(f"✅ {self.operation}: No errors")
            return
        
        error_types = self.get_error_types()
        logger.warning(
            f"❌ {self.operation}: {self.error_count()} error(s) "
            f"({', '.join(error_types)})"
        )
        
        for error_dict in self.errors:
            logger.debug(
                f"  - [{error_dict['type']}] {error_dict['message']} "
                f"({error_dict['context']})"
            )
    
    def log_detailed(self) -> None:
        """Log detailed error information"""
        logger.error("\n" + "=" * 80)
        logger.error(f"ERROR REPORT: {self.operation}")
        logger.error("=" * 80)
        logger.error(f"Timestamp: {self.timestamp.isoformat()}")
        logger.error(f"Total Errors: {self.error_count()}")
        logger.error(f"Error Types: {', '.join(self.get_error_types())}")
        
        for i, error_dict in enumerate(self.errors, 1):
            logger.error(f"\nError {i}:")
            logger.error(f"  Type: {error_dict['type']}")
            logger.error(f"  Message: {error_dict['message']}")
            logger.error(f"  Context: {error_dict['context']}")
            logger.error(f"  Timestamp: {error_dict['timestamp'].isoformat()}")
        
        logger.error("=" * 80 + "\n")


class ErrorReporter:
    """Central error reporting system"""
    
    def __init__(self):
        self.reports: List[ErrorReport] = []
        self.total_errors = 0
    
    def create_report(self, operation: str) -> ErrorReport:
        """Create a new error report"""
        report = ErrorReport(operation)
        self.reports.append(report)
        return report
    
    def add_error_to_report(
        self,
        report: ErrorReport,
        error: Exception,
        context: str = ""
    ) -> None:
        """Add error to a report and update counter"""
        report.add_error(error, context)
        self.total_errors += 1
    
    def get_critical_errors(self) -> List[Exception]:
        """Get all execution and API errors"""
        critical = []
        for report in self.reports:
            for error_dict in report.errors:
                error = error_dict['error_object']
                if isinstance(error, (ExecutionError, APIDataError)):
                    critical.append(error)
        return critical
    
    def get_validation_errors(self) -> List[Exception]:
        """Get all validation errors"""
        validation = []
        for report in self.reports:
            for error_dict in report.errors:
                error = error_dict['error_object']
                if isinstance(error, ValidationError):
                    validation.append(error)
        return validation
    
    def get_strategy_errors(self) -> List[Exception]:
        """Get all strategy errors"""
        strategy = []
        for report in self.reports:
            for error_dict in report.errors:
                error = error_dict['error_object']
                if isinstance(error, StrategyError):
                    strategy.append(error)
        return strategy
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of errors by type"""
        summary = {
            'total_errors': self.total_errors,
            'validation_errors': len(self.get_validation_errors()),
            'strategy_errors': len(self.get_strategy_errors()),
            'position_errors': len(self.get_position_errors()),
            'execution_errors': len(self.get_critical_errors()),
        }
        return summary
    
    def get_position_errors(self) -> List[Exception]:
        """Get all position errors"""
        position = []
        for report in self.reports:
            for error_dict in report.errors:
                error = error_dict['error_object']
                if isinstance(error, PositionError):
                    position.append(error)
        return position
    
    def log_session_summary(self) -> None:
        """Log summary of all errors in session"""
        summary = self.get_summary()
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 ERROR SESSION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Errors: {summary['total_errors']}")
        logger.info(f"  - Validation: {summary['validation_errors']}")
        logger.info(f"  - Strategy: {summary['strategy_errors']}")
        logger.info(f"  - Position: {summary['position_errors']}")
        logger.info(f"  - Execution: {summary['execution_errors']}")
        logger.info("=" * 80 + "\n")
    
    def clear(self) -> None:
        """Clear all reports"""
        self.reports.clear()
        self.total_errors = 0


# Global error reporter instance
_global_reporter = ErrorReporter()


def get_error_reporter() -> ErrorReporter:
    """Get global error reporter"""
    return _global_reporter
