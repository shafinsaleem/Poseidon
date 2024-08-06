package main

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"go.dedis.ch/kyber/v3/suites"
	"go.dedis.ch/onet/v3"
	"go.dedis.ch/onet/v3/log"
)

func uploadFile(w http.ResponseWriter, r *http.Request) {
	log.Lvl1("Received upload request")
	if r.Method != "POST" {
		log.Error("Invalid request method")
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		log.Error("Error retrieving file from request: ", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	log.Lvl1("Processing file: ", header.Filename)

	// Create uploads directory if it doesn't exist
	os.MkdirAll("uploads", os.ModePerm)

	// Create a temporary file to save the uploaded file
	tempFile, err := os.Create(filepath.Join("uploads", header.Filename))
	if err != nil {
		log.Error("Error creating temporary file: ", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer tempFile.Close()

	// Copy the uploaded file to the temporary file
	if _, err := io.Copy(tempFile, file); err != nil {
		log.Error("Error copying file: ", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	log.Lvl1("File uploaded successfully: ", tempFile.Name())

	// Execute the Python script with the uploaded file
	cmd := exec.Command("python3.7", "../predict_digit.py", tempFile.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Error("Error executing script: ", err)
		http.Error(w, fmt.Sprintf("Error executing script: %v", err), http.StatusInternalServerError)
		return
	}

	log.Lvl1("Script executed successfully")
	fmt.Fprintf(w, "Script output: %s", output)
}

func main() {
	// Initialize Suite
	log.Lvl1("Initializing suite")
	suite := suites.MustFind("Ed25519")

	// Load CA certificate
	log.Lvl1("Loading CA certificate")
	caCert, err := ioutil.ReadFile("ca.crt")
	if err != nil {
		log.Fatal("Failed to read CA certificate:", err)
	}
	caCertPool := x509.NewCertPool()
	caCertPool.AppendCertsFromPEM(caCert)

	// Load server certificate and key
	log.Lvl1("Loading server certificate and key")
	cert, err := tls.LoadX509KeyPair("server.crt", "server.key")
	if err != nil {
		log.Fatal("Failed to load server certificate and key:", err)
	}

	// Configure TLS
	log.Lvl1("Configuring TLS")
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientCAs:    caCertPool,
		ClientAuth:   tls.NoClientCert,
	}

	// Initialize Onet with custom TLS configuration
	log.Lvl1("Initializing Onet with TLS configuration")
	local := onet.NewLocalTest(suite)
	defer local.CloseAll()

	// Start HTTP server with TLS
	go func() {
		http.HandleFunc("/upload", uploadFile)

		httpServer := &http.Server{
			Addr:      ":2000",
			TLSConfig: tlsConfig,
		}
		log.Lvl1("Starting HTTP server with TLS on :2000")
		if err := httpServer.ListenAndServeTLS("server.crt", "server.key"); err != nil {
			log.Fatal("Failed to start HTTP server:", err)
		}
	}()

	log.Lvl1("Server started on :2000")

	// Create a channel to listen for interrupt signals
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	// Wait for an interrupt
	<-stop

	// Gracefully shut down the server
	log.Lvl1("Shutting down the server...")
	// No need to call server.Close() as we are using http.Server

	// Give the server a chance to shut down gracefully
	time.Sleep(1 * time.Second)
	log.Lvl1("Server shut down successfully")
}
