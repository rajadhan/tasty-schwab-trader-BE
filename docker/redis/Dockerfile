# Use the official Redis image from Docker Hub
FROM redis:latest

# Expose the default Redis port
EXPOSE 6379

# Run Redis with persistence enabled
CMD ["redis-server", "--save", "60", "1", "--loglevel", "warning"]
