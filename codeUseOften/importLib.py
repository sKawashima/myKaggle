version: "3.2"
services:
  data_analysis:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - 8888:8888
    volumes:
      - .:/data_analysis
