package engine

import (
	"fmt"
	"log/slog"
	"mango_truth/core/pkg"
	"mango_truth/core/pkg/modules"
	"time"
)

type MangoEngine struct {
	feed        <-chan any
	compute     chan<- modules.DetectionRequest
	computeResp <-chan modules.DetectionStatus
	storage     *Storage
	cfg         pkg.EngineConfig
}

func NewMangoEngine(cfg pkg.EngineConfig, feed <-chan any) (*MangoEngine, chan modules.DetectionRequest, chan modules.DetectionStatus) {
	computeReq := make(chan modules.DetectionRequest, cfg.ComputeBufferSize)
	computeResp := make(chan modules.DetectionStatus, cfg.ComputeBufferSize)

	ret := MangoEngine{
		feed:        feed,
		compute:     computeReq,
		computeResp: computeResp,
		storage:     &Storage{},
		cfg:         cfg}

	return &ret, computeReq, computeResp
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

		case <-time.After(time.Second * time.Duration(e.cfg.IdlePeriodSeconds)):
			slog.Debug("Engine idling..")
		}
	}
}
