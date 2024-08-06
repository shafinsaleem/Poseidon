package main

import (
	"os"
	"os/signal"
	"syscall"

	"go.dedis.ch/kyber/v3/suites"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
)

func main() {
	// Initialize Suite
	suite := suites.MustFind("Ed25519")

	// Initialize Onet
	local := onet.NewLocalTest(suite)
	defer local.CloseAll()

	// Create a new server instance
	server := local.NewServer(suite, 2000)
	if server == nil {
		log.Fatal("Failed to create server")
	}

	log.Print("Server started on :2000")

	// Create a channel to listen for interrupt signals
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	// Wait for an interrupt
	<-stop

	// Gracefully shut down the server
	log.Print("Shutting down the server...")
	server.Close()
}
