---
entry_id: docker-compose-persistent-storage
title: Docker Compose Persistent Storage Options
tags: [docker, docker-compose, volumes, persistence, containers]
created: 2024-05-27
last_modified: 2024-05-27
status: active
---

# Docker Compose Persistent Storage Options

When working with Docker containers, one of the common challenges is ensuring data persistence. By default, any data written inside a container is lost when the container is removed (e.g., after `docker-compose down`). This entry covers different approaches to implementing persistent storage in Docker Compose.

## Storage Types

Docker offers two primary methods for implementing persistent storage:

### 1. Docker Volumes

Docker volumes are managed by Docker itself and stored in a part of the host filesystem that's managed by Docker (usually `/var/lib/docker/volumes/` on Linux). These volumes:

- Are completely managed by Docker
- Cannot be easily accessed from the host filesystem
- Provide better isolation from host filesystem changes
- Work well across different operating systems
- Are the preferred method for persistent data in Docker

### 2. Bind Mounts

Bind mounts link a container directory directly to a directory on the host machine. These:

- Can be accessed and modified by host processes
- Use an absolute or relative path on the host machine
- Are dependent on the host filesystem having a specific directory structure
- May introduce security issues if container processes can modify host files

## Implementation in Docker Compose

### Using Docker Volumes

```yaml
version: '3'
services:
  postgres:
    image: postgres:latest
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
    # Optional: specify driver and configuration
    # driver: local
    # driver_opts:
    #   type: none
    #   o: bind
    #   device: /custom/path/on/host  # To specify a custom location (advanced)
```

- This creates a named volume `postgres_data` managed by Docker
- Data persists between container restarts and removals
- The volume remains even after `docker-compose down` (requires `docker-compose down -v` to remove)

### Using Bind Mounts

```yaml
version: '3'
services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./website:/usr/share/nginx/html
      # Format: HOST_PATH:CONTAINER_PATH
```

- This mounts the `./website` directory from the host into the container
- Changes made by either the host or container are visible to both
- Relative paths are relative to the docker-compose.yml file location

## Best Practices

1. **For Application Data**: Use Docker volumes for databases, caches, and other application state to ensure proper performance and isolation.

2. **For Development**: Use bind mounts for source code during development to enable hot-reloading and direct editing.

3. **Naming Conventions**: Use descriptive names for volumes that indicate their purpose.

4. **Backup Strategy**: Implement a strategy to backup volume data regularly.

5. **Volume Cleanup**: Be mindful of orphaned volumes - use `docker volume prune` for cleanup.

6. **Initialization**: For databases, use initialization scripts placed in special directories recognized by the database image.

## Common Issues

- **Permission Problems**: Container processes may run as different user IDs than the host, causing permission issues. Often resolved with proper UID/GID mapping.

- **Path Differences**: Absolute paths in bind mounts must exist on the host system.

- **Deletion Persistence**: When files are deleted inside a container, and then the container is recreated from the image, the original image files may reappear alongside volume contents. This is because Docker applies volume contents on top of the image's filesystem.

## Examples

### MongoDB Persistence

```yaml
services:
  mongo:
    image: mongo:4.4
    environment:
      - MONGO_INITDB_ROOT_USERNAME=root
      - MONGO_INITDB_ROOT_PASSWORD=pass
    ports:
      - '27017:27017'
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

### Development Environment with Source Code Hot-Reloading

```yaml
services:
  app:
    build: .
    ports:
      - "3000:3000"
    volumes:
      - ./src:/app/src
      - node_modules:/app/node_modules

volumes:
  node_modules:
```

This entry provides an overview of Docker Compose persistence options to help developers effectively manage data in containerized applications.