# Use official Node.js light image
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json first for caching
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production

# Copy the rest of the application files
COPY . .

# Expose the internal port (default is 3000)
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
