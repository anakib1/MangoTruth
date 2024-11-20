package core

import (
	"fmt"
	"github.com/google/uuid"
	"log/slog"
	"mango_truth/pkg"
	"mango_truth/pkg/modules"
	"mango_truth/pkg/storage"
	"mango_truth/pkg/storage/models"
	"time"
)

type MangoEngine struct {
	feed        <-chan modules.ClientToServer
	computeSink chan<- modules.DetectionRequest
	computeFeed <-chan modules.DetectionStatus
	storage     *storage.Storage
	cfg         pkg.EngineConfig
}

func NewMangoEngine(enginsCfg pkg.EngineConfig, storage *storage.Storage) (
	engine *MangoEngine,
	computeSink chan modules.DetectionRequest,
	computeFeed chan modules.DetectionStatus,
	engineFeed chan modules.ClientToServer) {

	engineFeed = make(chan modules.ClientToServer, enginsCfg.FeedBufferSize)
	computeSink = make(chan modules.DetectionRequest, enginsCfg.ComputeBufferSize)
	computeFeed = make(chan modules.DetectionStatus, enginsCfg.ComputeBufferSize)

	engine = &MangoEngine{
		feed:        engineFeed,
		computeSink: computeSink,
		computeFeed: computeFeed,
		storage:     storage,
		cfg:         enginsCfg}
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
						Status:    models.StatusPENDING,
					}
					e.storage.UpdStatus(status)
					cts.Ret <- status

				case modules.DetectionQuery:
					status := e.storage.GetStatus(uuid.MustParse(req.RequestId))
					cts.Ret <- status

				case modules.MassDetectionStatusRequest:
					status := e.storage.MassStatus()
					cts.Ret <- status
				default:
					slog.Warn(fmt.Sprintf("Received message of unexepcted type. Type = %T", req))
				}
			}
		case upd := <-e.computeFeed:
			e.storage.UpdStatus(upd)

		case <-time.After(time.Second * time.Duration(e.cfg.IdlePeriodSeconds)):
			slog.Debug("Engine idling..")
		}
	}
}
