# Email OTP setup

Volunteer signup verifies the email before creating the account. Volunteer login also uses email OTP. Phone remains required on signup as a volunteer contact/profile field. NGO password authentication is unchanged.

## Configure Gmail SMTP

1. Choose a Gmail account to send CivicEngage messages from.
2. Enable **2-Step Verification** for that Google account.
3. In the Google Account's **Security** settings, create an **App Password** for CivicEngage. Google may hide this option for managed accounts or accounts with Advanced Protection; use an organization-approved SMTP provider if App Passwords are unavailable.
4. Copy the repository `.env.example` values into the repository-root `.env` (or `backend/.env`). Set `GMAIL_USER` to the sender account and `GMAIL_APP_PASSWORD` to the generated App Password. Do not use the normal Google account password.
5. Set `OTP_SECRET` to a long random secret. For example, generate 32 random bytes with Node's `crypto.randomBytes(32).toString('hex')` and paste the output into the local `.env`.
6. Keep the credentials only in `.env`. `.env` is git-ignored. Never add SMTP credentials to frontend variables such as `VITE_*`, commit them, or paste them into source code.

`GMAIL_FROM` is optional; if omitted, mail is sent as the configured Gmail account. The server loads the repository-root `.env` first and then `backend/.env`; the root value takes precedence if the same variable appears in both.

## Existing email data / unique index

The volunteer email index is unique for non-empty values. Before starting this version against an existing MongoDB database, inspect duplicate volunteer emails using this `mongosh` aggregation (replace `yourDatabase` with the database name):

```javascript
use yourDatabase
db.users.aggregate([
	{ $match: { email: { $type: "string", $ne: "" } } },
	{ $project: { normalizedEmail: { $toLower: { $trim: { input: "$email" } } } } },
	{ $group: { _id: "$normalizedEmail", count: { $sum: 1 }, userIds: { $push: "$_id" } } },
	{ $match: { count: { $gt: 1 } } }
])
```

Resolve any duplicates first, deciding which profile data to preserve. Then normalize the existing non-empty values before starting the updated backend:

```javascript
db.users.updateMany(
	{ email: { $type: "string", $ne: "" } },
	[{ $set: { email: { $toLower: { $trim: { input: "$email" } } } } }]
)
```

The index intentionally excludes empty strings so legacy accounts without an email can remain in the collection, but such users will need an email added before they can use OTP login.

## API and user flow

- `POST /api/auth/user/signup/send-otp` accepts the signup profile, including `name`, `phone`, and `email`. The backend stores a temporary challenge and sends a 6-digit code; it does not create the user yet.
- `POST /api/auth/user/signup/verify-otp` accepts `{ "email": "...", "otp": "123456" }`. A valid code is consumed once, the account is created from the server-held pending signup data, and the response contains the normal user JWT/profile for automatic frontend login.
- `POST /api/auth/user/login/send-otp` accepts `{ "email": "..." }`.
- `POST /api/auth/user/login/verify-otp` accepts `{ "email": "...", "otp": "123456" }` and returns the normal user JWT/profile.

Codes are hashed in MongoDB, expire after five minutes, allow five verification attempts, are invalidated after successful verification, and cannot be reused. Requests are rate-limited per IP, with a per-email one-minute resend cooldown. MongoDB TTL cleanup is asynchronous; verification checks expiry directly and does not rely on TTL timing. API responses never contain the code.

After configuring `.env`, start MongoDB and the backend, then start the frontend. Submit a new signup email and verify that the mail arrives before testing the code. If mail delivery fails, check the backend configuration/error response and confirm the account uses an App Password. Never log or share a real OTP or App Password when reporting a problem.