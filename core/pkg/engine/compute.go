package engine

import (
	"encoding/json"
	"fmt"
	amqp "github.com/rabbitmq/amqp091-go"
	"log/slog"
	"mango_truth/core/pkg/modules"
	"time"
)

type ComputeRouter struct {
	feed   <-chan modules.DetectionRequest
	sink   chan<- modules.DetectionStatus
	rCh    *amqp.Channel
	rSinkQ *amqp.Queue
	rFeedQ *amqp.Queue
	rFeed  <-chan amqp.Delivery
}

func mustExistQueue(ch *amqp.Channel, name string) *amqp.Queue {
	q, err := ch.QueueDeclare(name, false, false, false, false, nil)
	if err != nil {
		panic(fmt.Sprintf("Could not declare queue. Error:  %v", err))
	}
	return &q
}

func mustExistRabbitMQ(url string) *amqp.Connection {
	conn, err := amqp.Dial(url)
	if err != nil {
		panic(fmt.Sprintf("Could not connect to RabbitMQ. Error:  %v", err))
	}

	return conn
}

func mustExistChannel(conn *amqp.Connection) *amqp.Channel {
	ch, err := conn.Channel()
	if err != nil {
		panic(fmt.Sprintf("Could not open channel. Error:  %v", err))
	}
	return ch
}

func mustConsume(ch *amqp.Channel, feed *amqp.Queue) <-chan amqp.Delivery {
	m, err := ch.Consume(feed.Name, "", true, false, false, false, nil)
	if err != nil {
		panic(err)
	}
	return m
}

func NewComputeRouter(feed <-chan modules.DetectionRequest, sink chan<- modules.DetectionStatus) *ComputeRouter {
	conn := mustExistRabbitMQ("amqp://guest:guest@localhost:5672/")
	ch := mustExistChannel(conn)

	rabbitSink := mustExistQueue(ch, "DetectionRequests")
	rabbitFeed := mustExistQueue(ch, "DetectionResponses")

	rFeed := mustConsume(ch, rabbitFeed)

	return &ComputeRouter{feed: feed, sink: sink, rCh: ch, rSinkQ: rabbitSink, rFeedQ: rabbitFeed, rFeed: rFeed}
}

func (c *ComputeRouter) Work() {
	for {
		select {
		case msg := <-c.feed:
			slog.Debug("Compute feed",
				"msg", msg)
			req, err := json.Marshal(msg)
			if err != nil {
				c.sink <- modules.DetectionStatus{Status: "PARSING_FAILED"}
				continue
			}
			kerr := c.rCh.Publish("", c.rSinkQ.Name, false, false, amqp.Publishing{ContentType: "application/json", Body: req})
			if kerr != nil {
				c.sink <- modules.DetectionStatus{Status: "TRANSPORT_FAILED"}
			}
		case msg := <-c.rFeed:
			slog.Debug("Compute rFeed",
				"msg", msg)
			var resp modules.DetectionStatus
			err := json.Unmarshal(msg.Body, &resp)
			if err != nil {
				slog.Warn("Failed to unmarshal message.",
					"Error", err)
				continue
			}
			c.sink <- resp
		case <-time.After(5 * time.Second):
			slog.Debug("Compute idling..")
		}
	}
}
