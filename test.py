import asyncio
import pytest
from lab1 import CircuitBreaker, State, CircuitBreakerOpenException

# Допоміжні функції-імітації (моки)
async def mock_success():
    return "ok"

async def mock_fail():
    raise RuntimeError("External service error")

async def mock_slow():
    await asyncio.sleep(0.5)
    return "ok"


@pytest.mark.asyncio
async def test_successful_calls_keep_closed():
    """Успішні виклики тримають стан CLOSED."""
    cb = CircuitBreaker(failure_threshold=2)
    assert cb.state() == State.CLOSED
    res = await cb.call(mock_success)
    assert res == "ok"
    assert cb.state() == State.CLOSED


@pytest.mark.asyncio
async def test_transition_to_open_after_n_failures():
    """Перехід у OPEN після N послідовних збоїв."""
    cb = CircuitBreaker(failure_threshold=3)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(mock_fail)
        assert cb.state() == State.CLOSED

    # 3-й збій переводить у стан OPEN
    with pytest.raises(RuntimeError):
        await cb.call(mock_fail)
    assert cb.state() == State.OPEN


@pytest.mark.asyncio
async def test_timeout_counted_as_failure():
    """Таймаут окремого виклику зараховується як збій."""
    cb = CircuitBreaker(failure_threshold=1, timeout_per_call=0.1)

    with pytest.raises(asyncio.TimeoutError):
        await cb.call(mock_slow)
    assert cb.state() == State.OPEN


@pytest.mark.asyncio
async def test_open_blocks_calls_until_duration_expires():
    """Блокування викликів у стані OPEN до спливання інтервалу."""
    cb = CircuitBreaker(failure_threshold=1, open_state_duration=0.3)
    with pytest.raises(RuntimeError):
        await cb.call(mock_fail)
    assert cb.state() == State.OPEN

    # Спроба виклику одразу відхиляється
    with pytest.raises(CircuitBreakerOpenException):
        await cb.call(mock_success)

    # Очікуємо спливання часу блокування
    await asyncio.sleep(0.35)
    assert cb.state() == State.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_recovers_to_closed():
    """Успішні виклики у HALF_OPEN повертають стан у CLOSED."""
    cb = CircuitBreaker(
        failure_threshold=1,
        open_state_duration=0.2,
        half_open_max_calls=2
    )
    with pytest.raises(RuntimeError):
        await cb.call(mock_fail)

    await asyncio.sleep(0.25)
    assert cb.state() == State.HALF_OPEN

    # Виконуємо необхідну кількість пробних викликів
    await cb.call(mock_success)
    assert cb.state() == State.HALF_OPEN

    await cb.call(mock_success)
    assert cb.state() == State.CLOSED


@pytest.mark.asyncio
async def test_half_open_fails_to_open():
    """Збій під час HALF_OPEN повертає коло у стан OPEN."""
    cb = CircuitBreaker(
        failure_threshold=1,
        open_state_duration=0.2,
        half_open_max_calls=2
    )
    with pytest.raises(RuntimeError):
        await cb.call(mock_fail)

    await asyncio.sleep(0.25)
    assert cb.state() == State.HALF_OPEN

    # Перший же збій повертає в OPEN
    with pytest.raises(RuntimeError):
        await cb.call(mock_fail)
    assert cb.state() == State.OPEN