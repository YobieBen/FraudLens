"""
Event bus implementation for FraudLens.

Author: Yobie Benjamin
Date: 2026-02-28
"""

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, TypeVar

from loguru import logger

from fraudlens.events.types import Event, EventPriority

# Type for event handlers
EventHandler = Callable[[Event], Awaitable[None]]
T = TypeVar("T", bound=Event)


class EventBus:
    """
    Async event bus for loosely coupled component communication.
    
    Features:
    - Async event handlers
    - Priority-based execution
    - Middleware support
    - Error handling and recovery
    - Wildcard subscriptions
    """
    
    def __init__(self):
        """Initialize event bus."""
        self._handlers: dict[type[Event], list[tuple[EventHandler, int]]] = defaultdict(list)
        self._wildcard_handlers: list[tuple[EventHandler, int]] = []
        self._middleware: list[Callable] = []
        self._running = True
    
    def subscribe(
        self,
        event_type: type[T],
        handler: Callable[[T], Awaitable[None]],
        priority: int = 0,
    ) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async handler function
            priority: Handler priority (higher = executed first)
        """
        self._handlers[event_type].append((handler, priority))
        # Sort by priority (descending)
        self._handlers[event_type].sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Subscribed handler to {event_type.__name__} with priority {priority}")
    
    def subscribe_all(
        self,
        handler: EventHandler,
        priority: int = 0,
    ) -> None:
        """
        Subscribe to all events (wildcard subscription).
        
        Args:
            handler: Async handler function
            priority: Handler priority
        """
        self._wildcard_handlers.append((handler, priority))
        self._wildcard_handlers.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Subscribed wildcard handler with priority {priority}")
    
    def unsubscribe(
        self,
        event_type: type[T],
        handler: Callable[[T], Awaitable[None]],
    ) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Event type to unsubscribe from
            handler: Handler to remove
        """
        handlers = self._handlers.get(event_type, [])
        self._handlers[event_type] = [
            (h, p) for h, p in handlers if h != handler
        ]
        
        logger.debug(f"Unsubscribed handler from {event_type.__name__}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        Add middleware to event processing pipeline.
        
        Middleware receives (event, next) and must call next(event)
        to continue the chain.
        
        Args:
            middleware: Middleware function
        """
        self._middleware.append(middleware)
        logger.debug(f"Added middleware: {middleware.__name__}")
    
    async def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribed handlers.
        
        Handlers are executed in priority order. Wildcard handlers
        are executed after specific handlers.
        
        Args:
            event: Event to emit
        """
        if not self._running:
            logger.warning("Event bus is stopped, ignoring event")
            return
        
        logger.debug(f"Emitting event: {event.__class__.__name__} (id={event.id})")
        
        # Execute middleware
        processed_event = event
        for middleware in self._middleware:
            try:
                processed_event = await middleware(processed_event)
                if processed_event is None:
                    logger.debug(f"Middleware stopped event propagation: {event.id}")
                    return
            except Exception as e:
                logger.error(f"Middleware error: {e}")
                continue
        
        # Get specific handlers
        event_type = type(processed_event)
        specific_handlers = self._handlers.get(event_type, [])
        
        # Combine specific and wildcard handlers
        all_handlers = specific_handlers + self._wildcard_handlers
        
        if not all_handlers:
            logger.debug(f"No handlers for event: {event_type.__name__}")
            return
        
        # Execute handlers based on priority
        tasks = []
        for handler, priority in all_handlers:
            task = self._execute_handler(handler, processed_event, priority)
            tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any errors
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Handler error: {result}")
    
    async def emit_many(self, events: list[Event]) -> None:
        """
        Emit multiple events.
        
        Events are processed sequentially to maintain order.
        
        Args:
            events: List of events to emit
        """
        for event in events:
            await self.emit(event)
    
    async def emit_and_wait(
        self,
        event: Event,
        timeout: float | None = None,
    ) -> list[Any]:
        """
        Emit event and wait for all handlers to complete.
        
        Args:
            event: Event to emit
            timeout: Optional timeout in seconds
        
        Returns:
            List of handler results
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        
        if not handlers:
            return []
        
        tasks = [
            self._execute_handler(handler, event, priority)
            for handler, priority in handlers
        ]
        
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _execute_handler(
        self,
        handler: EventHandler,
        event: Event,
        priority: int,
    ) -> Any:
        """
        Execute a single event handler with error handling.
        
        Args:
            handler: Handler function
            event: Event to handle
            priority: Handler priority
        
        Returns:
            Handler result or None if error
        """
        try:
            result = await handler(event)
            logger.trace(f"Handler executed successfully: {handler.__name__}")
            return result
        except Exception as e:
            logger.error(
                f"Error in event handler {handler.__name__} "
                f"for event {event.__class__.__name__}: {e}"
            )
            return None
    
    def clear(self) -> None:
        """Clear all handlers and middleware."""
        self._handlers.clear()
        self._wildcard_handlers.clear()
        self._middleware.clear()
        logger.info("Event bus cleared")
    
    def stop(self) -> None:
        """Stop the event bus."""
        self._running = False
        logger.info("Event bus stopped")
    
    def start(self) -> None:
        """Start the event bus."""
        self._running = True
        logger.info("Event bus started")
    
    def get_handler_count(self) -> int:
        """Get total number of registered handlers."""
        specific_count = sum(len(handlers) for handlers in self._handlers.values())
        return specific_count + len(self._wildcard_handlers)


# Global event bus instance
_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        Global EventBus instance
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def set_event_bus(bus: EventBus) -> None:
    """
    Set the global event bus instance.
    
    Args:
        bus: EventBus instance to set as global
    """
    global _global_bus
    _global_bus = bus
