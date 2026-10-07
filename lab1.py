import asyncio
import time
from enum import Enum
from typing import Any, Callable, Coroutine


class State(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenException(Exception):
    """Викидається, коли коло розімкнене і запити блокуються."""
    pass


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        half_open_max_calls: int = 2,
        open_state_duration: float = 2.0,  # час у секундах
        timeout_per_call: float = 1.0       # таймаут на 1 виклик у секундах
    ):
        self.failure_threshold = failure_threshold
        self.half_open_max_calls = half_open_max_calls
        self.open_state_duration = open_state_duration
        self.timeout_per_call = timeout_per_call

        self._state = State.CLOSED
        self._failure_count = 0
        self._last_state_change = time.monotonic()
        self._half_open_successful_calls = 0
        self._half_open_calls_in_flight = 0

    def state(self) -> State:
        """Повертає поточний стан, враховуючи час відновлення."""
        if self._state == State.OPEN:
            if time.monotonic() - self._last_state_change >= self.open_state_duration:
                self._transition_to(State.HALF_OPEN)
        return self._state

    def _transition_to(self, new_state: State):
        self._state = new_state
        self._last_state_change = time.monotonic()

        if new_state == State.CLOSED:
            self._failure_count = 0
            self._half_open_successful_calls = 0
            self._half_open_calls_in_flight = 0
        elif new_state == State.HALF_OPEN:
            self._half_open_successful_calls = 0
            self._half_open_calls_in_flight = 0
        elif new_state == State.OPEN:
            self._failure_count = 0

    async def call(self, fn: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs) -> Any:
        """Виконує функцію з контролем таймауту та стану."""
        current_state = self.state()

        # 1. Якщо коло розімкнене — швидкий відхил
        if current_state == State.OPEN:
            raise CircuitBreakerOpenException("Circuit is OPEN. Fast fail.")

        # 2. Якщо коло напіврозімкнене — обмеження кількості запитів
        if current_state == State.HALF_OPEN:
            if self._half_open_calls_in_flight >= self.half_open_max_calls:
                raise CircuitBreakerOpenException("Circuit is HALF_OPEN. Trial limit reached.")
            self._half_open_calls_in_flight += 1

        # 3. Виконання запиту з контролем таймауту
        try:
            result = await asyncio.wait_for(fn(*args, **kwargs), timeout=self.timeout_per_call)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        if self.state() == State.HALF_OPEN:
            self._half_open_successful_calls += 1
            if self._half_open_successful_calls >= self.half_open_max_calls:
                self._transition_to(State.CLOSED)
        else:
            self._failure_count = 0

    def _on_failure(self):
        if self.state() == State.HALF_OPEN:
            self._transition_to(State.OPEN)
        elif self.state() == State.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._transition_to(State.OPEN)