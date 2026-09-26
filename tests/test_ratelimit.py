from app.ratelimit import TokenBucket


def test_bucket_allows_burst_then_limits_then_refills():
    now = [0.0]
    b = TokenBucket(capacity=3, refill_per_s=1.0, clock=lambda: now[0])
    assert [b.try_acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert b.try_acquire() == 1.0  # empty: one token takes 1s to refill
    now[0] = 1.0
    assert b.try_acquire() == 0.0


def test_bucket_never_exceeds_capacity():
    now = [0.0]
    b = TokenBucket(capacity=2, refill_per_s=1.0, clock=lambda: now[0])
    now[0] = 100.0
    assert [b.try_acquire() for _ in range(3)][:2] == [0.0, 0.0]
    assert b.try_acquire() > 0
