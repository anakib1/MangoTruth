package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"mango_truth/pkg"
	"mango_truth/pkg/modules"
	"mango_truth/pkg/storage/models"

	"github.com/google/uuid"
	_ "github.com/lib/pq"
	"github.com/volatiletech/null/v8"
	"github.com/volatiletech/sqlboiler/v4/boil"
)

type Storage struct {
	db  boil.ContextExecutor
	cfg pkg.StorageConfig
}

func NewStorage(cfg pkg.StorageConfig) *Storage {
	db, err := sql.Open(cfg.DriverName, fmt.Sprintf("dbname=%s host=%s user=%s password=%s sslmode=disable", cfg.DatabaseName, cfg.HostName, cfg.UserName, cfg.Password))
	if err != nil {
		panic(fmt.Sprintf("Can not connect to the database: %s", err.Error()))
	}
	return &Storage{db: db, cfg: cfg}
}

func (s *Storage) UpdStatus(status modules.DetectionStatus) {
	data, err := json.Marshal(status.Verdict)
	if err != nil {
		panic(fmt.Sprintf("Can not connect verdict to json in DetectionStatus with requestId=%s, error: %s", status.RequestId, err.Error()))
	}
	new_status := models.DetectionStatus{
		RequestID: status.RequestId.String(),
		Status:    status.Status,
		Data:      null.BytesFrom(data),
	}

	err = new_status.Upsert(context.TODO(), s.db, true, []string{"request_id"}, boil.Infer(), boil.Infer())
	if err != nil {
		slog.Error("Can not insert value into storage", "error-msg", err.Error())
	}
}

func (s *Storage) GetStatus(id uuid.UUID) modules.DetectionStatus {
	status, err := models.FindDetectionStatus(context.TODO(), s.db, id.String())
	switch err {
	case nil:
		// Do nothing
	case sql.ErrNoRows:
		return modules.DetectionStatus{RequestId: id, Status: models.StatusUNKNOWN}
	default:
		slog.Error("Error in FindDetectionStatus", "error-msg", err.Error())
		return modules.DetectionStatus{RequestId: id, Status: models.StatusUNKNOWN}
	}
	request_id := uuid.MustParse(status.RequestID)
	var verdict modules.Verdict
	err = json.Unmarshal(status.Data.Bytes, &verdict)
	if err != nil {
		panic(fmt.Sprintf("Error parsing []bytes to Verdict, error: %s", err.Error()))
	}
	return modules.DetectionStatus{RequestId: request_id, Status: status.Status, Verdict: verdict}
}
