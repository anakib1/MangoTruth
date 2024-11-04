package utils

import (
	"errors"
	"log/slog"
	"mango_truth/core/pkg"
	"os"
	"strings"
)

func LogLevelByName(name string) (lev slog.Level, err error) {
	switch strings.ToLower(name) {
	case "debug":
		lev = slog.LevelDebug
	case "info":
		lev = slog.LevelInfo
	case "warn":
		lev = slog.LevelWarn
	case "error":
		lev = slog.LevelError
	default:
		err = errors.New("unknown level")
	}
	return
}

func ConfigureLogging(cfg *pkg.LoggerConfig) {
	lev, err := LogLevelByName(cfg.Level)
	if err != nil {
		println("Could not get log level correctly. Err : ", err.Error())
	}

	slog.SetLogLoggerLevel(lev)

	opts := &slog.HandlerOptions{Level: slog.LevelDebug}
	switch cfg.Format {
	case "json":
		slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, opts)))
	case "text":
		slog.SetDefault(slog.New(slog.NewTextHandler(os.Stdout, opts)))
	}
}
