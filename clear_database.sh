#!/bin/bash

# clear_database.sh
# Script to drop the neugenes MongoDB database

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}MongoDB Database Cleanup Script${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Check if mongosh is installed
if ! command -v mongosh &> /dev/null; then
    echo -e "${RED}Error: mongosh is not installed or not in PATH${NC}"
    echo "Please install MongoDB Shell: https://www.mongodb.com/try/download/shell"
    exit 1
fi

# Confirmation prompt
echo -e "${RED}WARNING: This will permanently delete the 'neugenes' database!${NC}"
echo -e "${YELLOW}All collections, documents, and GridFS files will be removed.${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo -e "${GREEN}Operation cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Dropping neugenes database...${NC}"

# Execute MongoDB commands
mongosh --quiet --eval "
    use neugenes;
    const result = db.dropDatabase();
    if (result.ok === 1) {
        print('✓ Database dropped successfully');
    } else {
        print('✗ Failed to drop database');
    }
"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Database cleared successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Also clean up local uploads directory
    echo ""
    read -p "Do you also want to clear the local uploads directory? (yes/no): " clear_uploads
    
    if [ "$clear_uploads" = "yes" ]; then
        UPLOADS_DIR="./server/uploads"
        if [ -d "$UPLOADS_DIR" ]; then
            echo -e "${YELLOW}Clearing uploads directory...${NC}"
            rm -rf "$UPLOADS_DIR"/*
            echo -e "${GREEN}✓ Uploads directory cleared${NC}"
        else
            echo -e "${YELLOW}Uploads directory not found at $UPLOADS_DIR${NC}"
        fi
    fi
    
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Error: Failed to clear database${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Cleanup complete!${NC}"
