# Frontend image: build the Vite app, serve the static bundle with nginx.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
# API base is same-origin /api; nginx proxies it to the backend service.
ENV VITE_API_BASE_URL=""
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
