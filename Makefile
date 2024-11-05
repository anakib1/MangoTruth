network:
	docker network create -d bridge mango-net

run-rabbitmq:
	docker image pull rabbitmq:4.0-management
	docker run -d --network mango-net --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4.0-management

run-core:
	docker build --tag mango-truth-core:1.0.0 --file ./core/Dockerfile .
	docker run -d --network mango-net --name core -p 8080:8080 -e COMPUTE_HOST='rabbitmq' mango-truth-core:1.0.0
