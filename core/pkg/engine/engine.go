package engine

import (
	"fmt"
	"log/slog"
	"mango_truth/core/pkg/modules"
	"time"
)

type MangoEngine struct {
	feed        <-chan any
	compute     chan<- modules.DetectionRequest
	computeResp <-chan modules.DetectionStatus
	storage     *Storage
}

func NewMangoEngine(feed <-chan any) *MangoEngine {
	computeReq := make(chan modules.DetectionRequest)
	computeResp := make(chan modules.DetectionStatus)
	go NewComputeRouter(computeReq, computeResp).Work()

	ret := MangoEngine{feed: feed, compute: computeReq, computeResp: computeResp, storage: &Storage{}}

	return &ret
}

func (e *MangoEngine) Work() {
	for {
		select {
		case msg := <-e.feed:
			{
				slog.Debug("Engine feed",
					"msg", msg)
				switch req := msg.(type) {
				case modules.DetectionRequestEx:
					e.compute <- req.DetectionRequest
					status := modules.DetectionStatus{
						BaseRequest: modules.BaseRequest{RequestId: req.RequestId},
						Status:      "PENDING",
					}
					e.storage.updStatus(status)
					req.PerformCallback(status)

				case modules.DetectionQueryEx:
					status := e.storage.getStatus(req.RequestId)
					req.PerformCallback(status)
				default:
					slog.Warn(fmt.Sprintf("Received message of unexepcted type. Type = %T", req))
				}
			}
		case upd := <-e.computeResp:
			e.storage.updStatus(upd)

		case <-time.After(5 * time.Second):
			slog.Debug("Engine idling..")
		}
	}
}
