# DrowseGuard — MongoDB Atlas + `.env`

This version keeps the existing DrowseGuard UI and drowsiness-detection logic, but the MongoDB connection is loaded from `.env` instead of being hard-coded.

## 1. Configure MongoDB Atlas

1. Create a MongoDB Atlas cluster.
2. Create a database user.
3. Add your IP address under **Network Access** (for local testing, you can temporarily allow `0.0.0.0/0`; restrict it for production when possible).
4. Copy the Atlas connection string.

## 2. Create `.env`

Copy `.env.example` to `.env` and replace the placeholders:

```env
MONGO_URL=mongodb+srv://USERNAME:PASSWORD@YOUR-CLUSTER.mongodb.net/drowsiness_db?retryWrites=true&w=majority
MONGO_DB_NAME=drowsiness_db
```

Do not commit `.env` to Git. It is already included in `.gitignore`.

If your MongoDB password contains characters such as `@`, `:`, `/`, `?`, or `#`, URL-encode the password before putting it in the connection string.

## 3. Install dependencies

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 4. Start the application

```bash
uvicorn server:app --reload
```

Open:

```text
http://localhost:8000
```

The frontend now uses `window.location.origin`, so it does not contain a hard-coded `http://localhost:8000` API URL.

## MongoDB collections

The application uses the configured database and creates/uses:

- `drivers`
- `sessions`
- `alerts`

Driver creation, sessions, and warning/danger events are persisted in MongoDB Atlas.

## Deployment

For a hosting provider, do **not** upload `.env` if the provider supports environment variables. Add:

- `MONGO_URL`
- `MONGO_DB_NAME`

The application loads `.env` locally through `python-dotenv` and reads the same variables from the deployment environment in production.
