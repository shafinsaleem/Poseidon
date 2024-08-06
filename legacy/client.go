package main

import (
	"os"

	"go.dedis.ch/kyber/v3/suites"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
	"go.dedis.ch/onet/v3/network"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run client.go <server-address>")
	}

	// Define the server address
	serverAddress := os.Args[1]

	// Initialize Suite
	suite := suites.MustFind("Ed25519")

	// Initialize Client
	client := onet.NewClient(suite, "poseidon_mininet")

	// Define the server identity
	address := network.NewAddress(network.PlainTCP, serverAddress)
	serverIdentity := &network.ServerIdentity{Address: address}

	// Example message structure
	type ExampleMessage struct {
		Content string
	}

	// Create the message
	message := ExampleMessage{Content: "Hello, Server!"}

	// Send the message using SendProtobuf
	err := client.SendProtobuf(serverIdentity, &message, nil)
	if err != nil {
		log.Fatalf("Failed to send message: %v", err)
	} else {
		log.Print("Message sent to server")
	}
}
