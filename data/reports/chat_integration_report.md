# Chat integration test

Run 2026-10-31T22:31:59+0800 by `python scripts/chat_integration_test.py` against a copy of `car_rental_db` (`car_rental_db_chat_it`, dropped afterwards), the PHP Back-End on port 3100 and the Django AI service on port 8101.

**57 of 57 chat checks passed.** Environment warnings: none.

| Check | Result |
|---|---|
| database copied to car_rental_db_chat_it | pass |
| users.role accepts the application's 'customer' role (migration 007 applied) | pass |
| Django AI service started | pass |
| PHP Back-End started (php -S, as php lava serve does) | pass |
| unauthenticated chat is rejected with 401 | pass |
| customer_a signs in | pass |
| customer_b signs in | pass |
| admin signs in | pass |
| price question answered from MySQL | pass |
| Taglish follow-up uses the same vehicle and the reservation pricing rule | pass |
| make-only price question lists that make's cars | pass |
| availability lists the available cars | pass |
| a changed price is used on the next question | pass |
| a newly added vehicle is found without any code change | pass |
| customer A sees their own reservation | pass |
| customer A does not see customer B's reservation | pass |
| customer B sees only their own reservation | pass |
| an id written in the message does not change whose bookings are read | pass |
| customer request for everyone's reservations is refused | pass |
| prompt injection does not unlock administrator data | pass |
| administrator sees every reservation | pass |
| administrator answer has no customer emails | pass |
| administrator maintenance records come from MySQL | pass |
| administrator due-for-service answer lists the overdue test car | pass |
| fleet count matches the cars table | pass |
| registration creates an account | pass |
| a registered account gets the customer role | pass |
| the new customer can sign in | pass |
| profile with licence details and photo is saved | pass |
| licence number, expiry and photo are stored | pass |
| a customer can view their own verification photo | pass |
| another customer cannot get that photo (only their own, which does not exist) | pass |
| signed-out requests cannot get a verification photo | pass |
| an expired licence is rejected | pass |
| the shared service key is configured (32+ characters) | pass |
| the AI service rejects a request without the key | pass |
| the AI service rejects a wrong key | pass |
| the AI service accepts the shared key | pass |
| POST /api/chat without a trailing slash is not a server error | pass |
| POST /api/predict without a trailing slash is not a server error | pass |
| POST /api/fleet/analyze without a trailing slash is not a server error | pass |
| POST /api/knowledge/search without a trailing slash is not a server error | pass |
| the AI service runs with DEBUG off (no debug error pages) | pass |
| administrator diagnostic mode answers for the selected vehicle | pass |
| customers cannot use diagnostic mode | pass |
| administrator mileage ranking | pass |
| administrator pending reservations from MySQL | pass |
| administrator status of one vehicle | pass |
| administrator maintenance cost total from MySQL | pass |
| two questions in one message are both answered | pass |
| 8 concurrent requests from two customers each see only their own bookings | pass |
| rate limiting stops a user after 20 chat messages a minute | pass |
| empty message is rejected with 422 | pass |
| malformed JSON is rejected without a server error | pass |
| a database error is reported, not answered from stale data | pass |
| an unavailable AI service returns 503 with a safe message | pass |
| the chat recovers when the AI service is back | pass |
| React frontend builds (output written to a temporary folder) | pass |
