# Bugs Found During Implementation

## 1. Commit-time conflict transitions were rejected

- **What broke:** A `slot_taken` or `hold_expired` response during `COMMITTING`
  raised an invalid-transition error instead of reaching a recoverable outcome.
- **Why:** The transition table only allowed those outcomes from `HOLDING`.
- **Fix:** Added `CONFLICT` and `EXPIRED` as valid `COMMITTING` transitions.
- **Regression:** `tests/test_evals.py` runs both slot-taken and hold-expired
  scenarios through the generated eval suite.

## 2. Timeout-after-success could be mistaken for failure

- **What broke:** A provider timeout did not tell us whether the booking had
  already committed.
- **Why:** The first implementation treated the transport error as the result.
- **Fix:** `BookingCoordinator` calls `get_booking_status` with the same
  idempotency key before retrying or reporting failure.
- **Regression:** `test_timeout_after_success_is_reconciled_without_retrying`.

## 3. Changed quotes could reuse an old approval

- **What broke:** A hold quote could differ from the search quote after the user
  approved the original price.
- **Why:** Approval was initially treated as a generic confirmation rather than
  an exact action binding.
- **Fix:** Approval hashes include provider, slot, operation, and price; quote
  changes update state and require a new approval.
- **Regression:** `test_quote_change_invalidates_old_approval_and_requests_new_confirmation`.
