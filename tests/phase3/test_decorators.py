"""Tests for observability decorators."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from fraudlens.observability.decorators import (
    counted,
    logged,
    observable,
    timed,
    traced,
)


class TestTracedDecorator:
    """Test @traced decorator."""
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.trace_operation")
    async def test_traced_async(self, mock_trace_op):
        """Test @traced with async function."""
        mock_trace_op.return_value.__enter__ = Mock()
        mock_trace_op.return_value.__exit__ = Mock()
        
        @traced()
        async def test_func(x: int) -> int:
            return x * 2
        
        result = await test_func(5)
        
        assert result == 10
        mock_trace_op.assert_called_once()
    
    @patch("fraudlens.observability.decorators.trace_operation")
    def test_traced_sync(self, mock_trace_op):
        """Test @traced with sync function."""
        mock_trace_op.return_value.__enter__ = Mock()
        mock_trace_op.return_value.__exit__ = Mock()
        
        @traced()
        def test_func(x: int) -> int:
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        mock_trace_op.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.trace_operation")
    async def test_traced_with_exception(self, mock_trace_op):
        """Test @traced propagates exception."""
        mock_trace_op.return_value.__enter__ = Mock()
        mock_trace_op.return_value.__exit__ = Mock(return_value=False)  # Don't suppress exception
        
        @traced()
        async def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await test_func()


class TestTimedDecorator:
    """Test @timed decorator."""
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    async def test_timed_async(self, mock_get_metrics):
        """Test @timed with async function."""
        mock_collector = Mock()
        mock_latency = Mock()
        mock_collector.request_latency_seconds = mock_latency
        mock_get_metrics.return_value = mock_collector
        
        @timed(metric_name="test_op")
        async def test_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2
        
        result = await test_func(5)
        
        assert result == 10
        # Should have called labels and observe
        assert mock_latency.labels.called
    
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    def test_timed_sync(self, mock_get_metrics):
        """Test @timed with sync function."""
        mock_collector = Mock()
        mock_latency = Mock()
        mock_collector.request_latency_seconds = mock_latency
        mock_get_metrics.return_value = mock_collector
        
        @timed(metric_name="test_op")
        def test_func(x: int) -> int:
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        assert mock_latency.labels.called


class TestCountedDecorator:
    """Test @counted decorator."""
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    async def test_counted_async(self, mock_get_metrics):
        """Test @counted with async function."""
        mock_collector = Mock()
        mock_counter = Mock()
        mock_collector.fraud_detections_total = mock_counter
        mock_get_metrics.return_value = mock_collector
        
        @counted(counter_name="test_calls")
        async def test_func(x: int) -> int:
            return x * 2
        
        result = await test_func(5)
        
        assert result == 10
        # Should have called labels and inc
        assert mock_counter.labels.called
    
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    def test_counted_sync(self, mock_get_metrics):
        """Test @counted with sync function."""
        mock_collector = Mock()
        mock_counter = Mock()
        mock_collector.fraud_detections_total = mock_counter
        mock_get_metrics.return_value = mock_collector
        
        @counted(counter_name="test_calls")
        def test_func(x: int) -> int:
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        assert mock_counter.labels.called
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    async def test_counted_with_error(self, mock_get_metrics):
        """Test @counted tracks errors."""
        mock_collector = Mock()
        mock_counter = Mock()
        mock_collector.fraud_detections_total = mock_counter
        mock_get_metrics.return_value = mock_collector
        
        @counted(counter_name="test_calls")
        async def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await test_func()
        
        # Should have called labels for error case
        assert mock_counter.labels.called


class TestLoggedDecorator:
    """Test @logged decorator."""
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.get_logger")
    async def test_logged_async(self, mock_get_logger):
        """Test @logged with async function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @logged()
        async def test_func(x: int) -> int:
            return x * 2
        
        result = await test_func(5)
        
        assert result == 10
        # Should have called log() method
        assert mock_logger.log.called
    
    @patch("fraudlens.observability.decorators.get_logger")
    def test_logged_sync(self, mock_get_logger):
        """Test @logged with sync function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @logged()
        def test_func(x: int) -> int:
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        assert mock_logger.log.called
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.get_logger")
    async def test_logged_with_exception(self, mock_get_logger):
        """Test @logged logs exception."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @logged()
        async def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await test_func()
        
        mock_logger.error.assert_called_once()


class TestObservableDecorator:
    """Test @observable all-in-one decorator."""
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.trace_operation")
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    @patch("fraudlens.observability.decorators.get_logger")
    async def test_observable_async(self, mock_get_logger, mock_get_metrics, mock_trace_op):
        """Test @observable combines all decorators."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_collector = Mock()
        mock_collector.request_latency_seconds = Mock()
        mock_collector.fraud_detections_total = Mock()
        mock_get_metrics.return_value = mock_collector
        mock_trace_op.return_value.__enter__ = Mock()
        mock_trace_op.return_value.__exit__ = Mock()
        
        @observable(component="test", agent_id="test_agent")
        async def test_func(x: int) -> int:
            return x * 2
        
        result = await test_func(5)
        
        assert result == 10
        # Verify all observability features were used
        assert mock_logger.log.called or mock_get_logger.called
        assert mock_collector.fraud_detections_total.labels.called
        assert mock_trace_op.called
    
    @patch("fraudlens.observability.decorators.trace_operation")
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    @patch("fraudlens.observability.decorators.get_logger")
    def test_observable_sync(self, mock_get_logger, mock_get_metrics, mock_trace_op):
        """Test @observable with sync function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_collector = Mock()
        mock_collector.request_latency_seconds = Mock()
        mock_collector.fraud_detections_total = Mock()
        mock_get_metrics.return_value = mock_collector
        mock_trace_op.return_value.__enter__ = Mock()
        mock_trace_op.return_value.__exit__ = Mock()
        
        @observable(component="test", agent_id="test_agent")
        def test_func(x: int) -> int:
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        assert mock_logger.log.called or mock_get_logger.called
        assert mock_collector.fraud_detections_total.labels.called


class TestDecoratorIntegration:
    """Test decorator combinations."""
    
    @pytest.mark.asyncio
    @patch("fraudlens.observability.decorators.get_metrics_collector")
    @patch("fraudlens.observability.decorators.get_logger")
    async def test_multiple_decorators(self, mock_get_logger, mock_get_metrics):
        """Test stacking multiple decorators."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_collector = Mock()
        mock_collector.request_latency_seconds = Mock()
        mock_collector.fraud_detections_total = Mock()
        mock_get_metrics.return_value = mock_collector
        
        @logged()
        @timed(metric_name="test")
        @counted(counter_name="test_calls")
        async def test_func(x: int) -> int:
            return x * 2
        
        result = await test_func(5)
        
        assert result == 10
        assert mock_logger.log.called
        assert mock_collector.fraud_detections_total.labels.called
        assert mock_collector.request_latency_seconds.labels.called
