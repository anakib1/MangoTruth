package core

import (
	"fmt"
	"log/slog"
	"mango_truth/core/pkg"
	"mango_truth/core/pkg/modules"
	"time"
)

type MangoEngine struct {
	feed        <-chan modules.ClientToServer
	computeSink chan<- modules.DetectionRequest
	computeFeed <-chan modules.DetectionStatus
	storage     *Storage
	cfg         pkg.EngineConfig
}

func NewMangoEngine(cfg pkg.EngineConfig) (
	engine *MangoEngine,
	computeSink chan modules.DetectionRequest,
	computeFeed chan modules.DetectionStatus,
	engineFeed chan modules.ClientToServer) {

	engineFeed = make(chan modules.ClientToServer, cfg.FeedBufferSize)
	computeSink = make(chan modules.DetectionRequest, cfg.ComputeBufferSize)
	computeFeed = make(chan modules.DetectionStatus, cfg.ComputeBufferSize)

	engine = &MangoEngine{
		feed:        engineFeed,
		computeSink: computeSink,
		computeFeed: computeFeed,
		storage:     &Storage{},
		cfg:         cfg}
	return
}

func (e *MangoEngine) Work() {
	for {
		select {
		case cts := <-e.feed:
			{
				slog.Debug("Engine feed",
					"msg", cts.Msg)
				switch req := cts.Msg.(type) {
				case modules.DetectionRequest:
					e.computeSink <- req
					status := modules.DetectionStatus{
						RequestId: req.RequestId,
						Status:    "PENDING",
					}
					e.storage.updStatus(status)
					cts.Ret <- status

				case modules.DetectionQuery:
					status := e.storage.getStatus(req.RequestId)
					cts.Ret <- status
				default:
					slog.Warn(fmt.Sprintf("Received message of unexepcted type. Type = %T", req))
				}
			}
		case upd := <-e.computeFeed:
			e.storage.updStatus(upd)

		case <-time.After(time.Second * time.Duration(e.cfg.IdlePeriodSeconds)):
			slog.Debug("Engine idling..")
		}
	}
}
