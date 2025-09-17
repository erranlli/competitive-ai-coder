# Start FROM the official image
FROM ghcr.io/engineer-man/piston

# Set the working directory
WORKDIR /piston_api

# Copy our patched application and startup script
COPY ./index.js ./src/index.js
COPY ./entrypoint.sh ./src/entrypoint.sh

# Make our new startup script executable
RUN chmod +x ./src/entrypoint.sh

# Set our script as the command to run when the container starts
CMD ["/piston_api/src/entrypoint.sh"]
