# Ares Backend V0.1.0

- REST API for ares-frontend
- Integration with linuxCNC RPC server
- Integration with Keyence python API
- Database access layer
- SMB file transfer

## Installation

### Requirements
1) Python 3.7+
    - https://www.python.org/downloads/
2) pg_config (for psycopg2)

### Initial Install

Download source code
```bash
git clone https://bitbucket.org/skatescribe/ares-backend.git
```

Change directory
```bash
cd ares-backend
```

Run setup script
```bash
./scripts/setup.sh
```

### Updating Dependencies
You can either manually update the `requirements.txt` or run:
```bash
pip freeze > requirements.txt
```

## Usage

### Starting the server
```bash
./start.sh
```
