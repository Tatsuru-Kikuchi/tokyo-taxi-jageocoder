#!/usr/bin/env python3
"""
JAGeocoder Backend Service for Tokyo AI Taxi App - Fixed Version
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import jageocoder
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="JAGeocoder Backend for Tokyo AI Taxi",
    description="Precise Japanese address geocoding service",
    version="1.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
geocoder_initialized = False
initialization_error = None

def initialize_jageocoder(retries=3, delay=30):
    """Initialize JAGeocoder with retry logic and better error handling"""
    global geocoder_initialized, initialization_error

    db_path = Path('/app/jageocoder_data')

    for attempt in range(retries):
        try:
            logger.info(f"JAGeocoder initialization attempt {attempt + 1}/{retries}")

            # Create database directory
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Database directory: {db_path}")

            # Check if database already exists
            if any(db_path.glob('*.db*')):
                logger.info("Existing JAGeocoder database found")
                # Try to initialize with existing database
                jageocoder.init(db_dir=str(db_path))
                logger.info("JAGeocoder initialized with existing database")
                geocoder_initialized = True
                return True

            # Download and initialize database
            logger.info("No existing database found. Starting download...")
            logger.info("This process may take 10-15 minutes for initial setup")

            # Try different initialization methods
            try:
                # Method 1: Auto-download during init
                jageocoder.init(db_dir=str(db_path))
                geocoder_initialized = True
                logger.info("JAGeocoder database initialized successfully")
                return True

            except Exception as init_error:
                logger.warning(f"Direct init failed: {init_error}")

                # Method 2: Manual download then init
                logger.info("Attempting manual download...")
                import jageocoder.download

                # Download database
                jageocoder.download.download_and_install(db_dir=str(db_path))

                # Initialize after download
                jageocoder.init(db_dir=str(db_path))
                geocoder_initialized = True
                logger.info("JAGeocoder initialized after manual download")
                return True

        except Exception as e:
            error_msg = f"Initialization attempt {attempt + 1} failed: {e}"
            logger.error(error_msg)
            initialization_error = error_msg

            if attempt < retries - 1:
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                logger.error("All initialization attempts failed")

    return False

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Earth's radius in kilometers
    r = 6371

    return r * c

@app.on_event("startup")
async def startup_event():
    """Initialize JAGeocoder on startup"""
    logger.info("Starting JAGeocoder backend service...")

    # Start initialization in background
    logger.info("Initializing JAGeocoder database...")
    success = initialize_jageocoder()

    if success:
        logger.info("JAGeocoder service ready")
    else:
        logger.warning("JAGeocoder initialization failed - service running with limited functionality")

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    global geocoder_initialized, initialization_error

    status = {
        "status": "healthy" if geocoder_initialized else "degraded",
        "service": "JAGeocoder Backend",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        "jageocoder_available": True,
        "geocoder_initialized": geocoder_initialized
    }

    if initialization_error:
        status["initialization_error"] = initialization_error

    return status

@app.get("/geocode/{address}")
async def geocode_address(address: str):
    """Geocode a Japanese address to coordinates"""
    global geocoder_initialized

    if not geocoder_initialized:
        # Try to initialize if not already done
        if not initialize_jageocoder(retries=1, delay=5):
            raise HTTPException(
                status_code=503,
                detail="JAGeocoder service not initialized. Please try again later."
            )

    try:
        logger.info(f"Geocoding address: {address}")

        # Use JAGeocoder to search for the address
        results = jageocoder.search(address)

        if not results:
            # Return approximate coordinates for major cities if exact match fails
            fallback_coords = get_fallback_coordinates(address)
            if fallback_coords:
                return {
                    "address": address,
                    "latitude": fallback_coords[0],
                    "longitude": fallback_coords[1],
                    "confidence": 0.3,
                    "source": "fallback",
                    "note": "Approximate coordinates used due to geocoding failure"
                }

            raise HTTPException(status_code=404, detail=f"Address not found: {address}")

        # Get the best match (first result)
        result = results[0]

        return {
            "address": address,
            "matched_address": result['fullname'],
            "latitude": result['y'],
            "longitude": result['x'],
            "confidence": result.get('score', 1.0),
            "source": "jageocoder"
        }

    except Exception as e:
        logger.error(f"Geocoding error for {address}: {e}")

        # Try fallback coordinates
        fallback_coords = get_fallback_coordinates(address)
        if fallback_coords:
            return {
                "address": address,
                "latitude": fallback_coords[0],
                "longitude": fallback_coords[1],
                "confidence": 0.2,
                "source": "fallback_emergency",
                "error": str(e)
            }

        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")

def get_fallback_coordinates(address: str) -> Optional[Tuple[float, float]]:
    """Get approximate coordinates for major Japanese locations as fallback"""
    fallback_locations = {
        "東京": (35.6762, 139.6503),
        "名古屋": (35.1815, 136.9066),
        "大阪": (34.6937, 135.5023),
        "京都": (35.0116, 135.7681),
        "福岡": (33.5904, 130.4017),
        "札幌": (43.0642, 141.3469),
        "仙台": (38.2682, 140.8694),
        "広島": (34.3853, 132.4553),
        "新宿": (35.6896, 139.6917),
        "渋谷": (35.6580, 139.7016),
        "池袋": (35.7295, 139.7109),
        "品川": (35.6284, 139.7387),
        "上野": (35.7141, 139.7774)
    }

    # Check if address contains any of these locations
    for location, coords in fallback_locations.items():
        if location in address:
            logger.info(f"Using fallback coordinates for {location}")
            return coords

    return None

@app.get("/distance")
async def calculate_distance(
    from_lat: float = Query(..., description="Starting latitude"),
    from_lng: float = Query(..., description="Starting longitude"),
    to_lat: float = Query(..., description="Destination latitude"),
    to_lng: float = Query(..., description="Destination longitude")
):
    """Calculate distance between two coordinates"""
    try:
        # Calculate distance using Haversine formula
        distance_km = haversine_distance(from_lat, from_lng, to_lat, to_lng)

        # Estimate duration (assuming average speed of 25 km/h in urban areas)
        duration_minutes = round(distance_km * 2.4)  # More conservative estimate

        return {
            "distance_km": round(distance_km, 2),
            "duration_minutes": duration_minutes,
            "calculation_method": "haversine"
        }

    except Exception as e:
        logger.error(f"Distance calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Distance calculation failed: {str(e)}")

@app.get("/")
async def root():
    """Service information endpoint"""
    return {
        "service": "JAGeocoder Backend for Tokyo AI Taxi",
        "version": "1.1.0",
        "status": "running",
        "endpoints": [
            "/geocode/{address} - Convert address to coordinates",
            "/distance - Calculate distance between points",
            "/health - Health check"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
