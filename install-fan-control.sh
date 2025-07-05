#!/bin/bash
# Installation script for Adaptive Supermicro fan control

set -e

echo "Installing Adaptive Supermicro ARS210M Fan Control..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root"
    exit 1
fi

# Check for ipmitool
if ! command -v ipmitool &> /dev/null; then
    echo "ipmitool is required but not installed. Installing..."
    apt-get update && apt-get install -y ipmitool || yum install -y ipmitool
fi

# Check for Python 3 and numpy
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    exit 1
fi

# Install numpy using apt-get
echo "Installing Python dependencies..."
apt-get update
apt-get install -y python3-numpy python3-scipy

# Stop existing fan control if running
if systemctl is-active --quiet fan-control; then
    echo "Stopping existing fan-control service..."
    systemctl stop fan-control
    systemctl disable fan-control
fi

# Copy the adaptive fan control script
echo "Installing adaptive fan control script..."
cp fan-control.py /usr/local/bin/fan-control.py
chmod +x /usr/local/bin/fan-control.py

# Copy systemd service (use existing service name)
echo "Installing systemd service..."
cp fan-control.service /etc/systemd/system/

# Create directories
mkdir -p /var/log
mkdir -p /var/lib
touch /var/log/fan-control.log
chmod 644 /var/log/fan-control.log

# Reload systemd
systemctl daemon-reload

# Enable but don't start
systemctl enable fan-control.service

echo "Installation complete!"
echo ""
echo "The adaptive controller will learn your system's thermal dynamics over time."
echo "It starts with reasonable defaults and improves with experience."
echo ""
echo "To test the fan control manually:"
echo "  python3 /usr/local/bin/fan-control.py"
echo ""
echo "To start the service:"
echo "  systemctl start fan-control"
echo ""
echo "To check status:"
echo "  systemctl status fan-control"
echo "  journalctl -u fan-control -f"
echo ""
echo "The learned thermal model is saved at: /var/lib/thermal_model.pkl"
echo "Delete this file to reset learning."