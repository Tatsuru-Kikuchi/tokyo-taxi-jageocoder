#!/usr/bin/env python3
"""
JAGeocoder Backend Service for Tokyo AI Taxi App - v3.0
Fixed for FastAPI lifespan events and variable scoping
"""

import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
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
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global state
geocoder_initialized = False
initialization_error = None

def initialize_jageocoder_v3():
    """Initialize JAGeocoder with updated API for version 2.1.10"""
    global geocoder_initialized, initialization_error

    try:
        logger.info("Initializing JAGeocoder v2.1.10...")

        # Try simple initialization first (recommended for v2.1.10)
        jageocoder.init()

        # Test if geocoder is working
        test_result = jageocoder.search('東京駅')
        if test_result and len(test_result) > 0:
            geocoder_initialized = True
            logger.info("JAGeocoder initialized successfully with existing data")
            return True
        else:
            # If no results, the database might not be installed
            logger.warning("JAGeocoder initialized but no data found")

        # Try to install the database using the correct method for v2.1.10
        logger.info("Attempting to install JAGeocoder database...")

        try:
            # Import the correct installer module
            import jageocoder.install_dictionary

            # Install the dictionary data
            jageocoder.install_dictionary.install()

            # Re-initialize after installation
            jageocoder.init()

            # Test again
            test_result = jageocoder.search('東京駅')
            if test_result and len(test_result) > 0:
                geocoder_initialized = True
                logger.info("JAGeocoder database installed and initialized successfully")
                return True
            else:
                raise Exception("Database installation completed but geocoding still not working")

        except ImportError as import_error:
            logger.error(f"JAGeocoder installation module not found: {import_error}")
            initialization_error = f"Installation module not available: {import_error}"

            # Fall back to basic initialization
            try:
                jageocoder.init()
                geocoder_initialized = True
                logger.warning("JAGeocoder initialized in basic mode - may have limited data")
                return True
            except Exception as basic_error:
                logger.error(f"Basic initialization also failed: {basic_error}")
                initialization_error = f"All initialization methods failed: {basic_error}"
                return False

    except Exception as e:
        logger.error(f"JAGeocoder initialization failed: {e}")
        initialization_error = str(e)

        # Try basic init as last resort
        try:
            jageocoder.init()
            geocoder_initialized = True
            logger.warning("JAGeocoder initialized with basic method")
            return True
        except Exception as basic_error:
            logger.error(f"Fallback initialization failed: {basic_error}")
            initialization_error = f"Complete initialization failure: {basic_error}"

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

def get_fallback_coordinates(address: str) -> Optional[Tuple[float, float]]:
    """Get approximate coordinates for major Japanese locations as fallback"""
    # Enhanced fallback with more locations
    fallback_locations = {
        # Major cities
        "東京": (35.6762, 139.6503),
        "名古屋": (35.1815, 136.9066),
        "大阪": (34.6937, 135.5023),
        "京都": (35.0116, 135.7681),
        "福岡": (33.5904, 130.4017),
        "札幌": (43.0642, 141.3469),
        "仙台": (38.2682, 140.8694),
        "広島": (34.3853, 132.4553),

        # Tokyo area stations/districts
        "新宿": (35.6896, 139.6917),
        "渋谷": (35.6580, 139.7016),
        "池袋": (35.7295, 139.7109),
        "品川": (35.6284, 139.7387),
        "上野": (35.7141, 139.7774),
        "東京駅": (35.6812, 139.7671),
        "新宿駅": (35.6896, 139.6917),
        "渋谷駅": (35.6580, 139.7016),

        # Aichi Prefecture (your area)
        "名古屋駅": (35.170694, 136.881636),
        "栄": (35.168058, 136.908245),
        "金山": (35.143033, 136.900656),
        "千種": (35.166584, 136.931411),
        "大曽根": (35.184089, 136.928358),
        "春日井": (35.248091, 136.971592),

        # Other major stations
        "大阪駅": (34.7024, 135.4959),
        "京都駅": (34.9859, 135.7585),
        "横浜駅": (35.4657, 139.6222),
    }

    # Check if address contains any of these locations
    for location, coords in fallback_locations.items():
        if location in address:
            logger.info(f"Using fallback coordinates for {location}")
            return coords

    # If no specific match, try to extract city names
    city_keywords = ["市", "区", "町", "村"]
    for keyword in city_keywords:
        if keyword in address:
            # Default to Tokyo for unknown locations
            logger.info(f"Using default Tokyo coordinates for address containing {keyword}")
            return (35.6762, 139.6503)

    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan event handler"""
    # Startup
    logger.info("Starting JAGeocoder backend service v3.0...")

    # Initialize JAGeocoder
    success = initialize_jageocoder_v3()

    if success:
        logger.info("JAGeocoder service ready")
    else:
        logger.warning("JAGeocoder initialization failed - service running with fallback functionality")

    yield

    # Shutdown
    logger.info("JAGeocoder backend service shutting down...")

# FastAPI app initialization with lifespan
app = FastAPI(
    title="JAGeocoder Backend for Tokyo AI Taxi",
    description="Precise Japanese address geocoding service",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    global geocoder_initialized, initialization_error

    # Try a quick test if geocoder claims to be initialized
    working = False
    if geocoder_initialized:
        try:
            test_result = jageocoder.search('東京駅')
            working = test_result is not None and len(test_result) > 0
        except Exception as e:
            logger.warning(f"JAGeocoder test failed during health check: {e}")
            working = False

    status = {
        "status": "healthy" if working else "degraded",
        "service": "JAGeocoder Backend",
        "version": "3.0.0",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        "jageocoder_available": True,
        "geocoder_initialized": geocoder_initialized,
        "geocoder_working": working
    }

    if initialization_error:
        status["initialization_error"] = initialization_error

    return status

@app.get("/geocode/{address}")
async def geocode_address(address: str):
    """Geocode a Japanese address to coordinates"""
    global geocoder_initialized

    logger.info(f"Geocoding request for: {address}")

    # Try JAGeocoder first if available
    if geocoder_initialized:
        try:
            results = jageocoder.search(address)

            if results and len(results) > 0:
                result = results[0]
                logger.info(f"JAGeocoder found result: {result.get('fullname', address)}")

                return {
                    "address": address,
                    "matched_address": result.get('fullname', address),
                    "latitude": result['y'],
                    "longitude": result['x'],
                    "confidence": result.get('score', 1.0),
                    "source": "jageocoder"
                }
            else:
                logger.warning(f"JAGeocoder returned no results for: {address}")

        except Exception as e:
            logger.error(f"JAGeocoder error for {address}: {e}")

    # Fallback to coordinate lookup
    fallback_coords = get_fallback_coordinates(address)
    if fallback_coords:
        logger.info(f"Using fallback coordinates for: {address}")
        return {
            "address": address,
            "latitude": fallback_coords[0],
            "longitude": fallback_coords[1],
            "confidence": 0.7,
            "source": "fallback",
            "note": "Approximate coordinates - JAGeocoder not fully available"
        }

    # If all fails, return error
    raise HTTPException(
        status_code=404,
        detail=f"Address not found and no fallback available: {address}"
    )

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

        # Estimate duration based on distance and urban conditions
        if distance_km <= 2:
            duration_minutes = max(5, int(distance_km * 8))  # City center: ~8 min/km
        elif distance_km <= 10:
            duration_minutes = int(distance_km * 4)  # Suburban: ~4 min/km
        else:
            duration_minutes = int(distance_km * 2.5)  # Highway/long distance: ~2.5 min/km

        logger.info(f"Distance calculated: {distance_km:.2f}km, ~{duration_minutes} minutes")

        return {
            "distance_km": round(distance_km, 2),
            "duration_minutes": duration_minutes,
            "calculation_method": "haversine"
        }

    except Exception as e:
        logger.error(f"Distance calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Distance calculation failed: {str(e)}")

@app.get("/test/{query}")
async def test_geocoder(query: str):
    """Test endpoint to debug geocoder functionality"""
    global geocoder_initialized

    result = {
        "query": query,
        "geocoder_initialized": geocoder_initialized,
        "jageocoder_result": None,
        "fallback_result": None,
        "error": None
    }

    # Test JAGeocoder
    if geocoder_initialized:
        try:
            jageocoder_results = jageocoder.search(query)
            result["jageocoder_result"] = jageocoder_results
        except Exception as e:
            result["error"] = str(e)

    # Test fallback
    fallback = get_fallback_coordinates(query)
    if fallback:
        result["fallback_result"] = {"lat": fallback[0], "lng": fallback[1]}

    return result

@app.get("/")
async def root():
    """Service information endpoint"""
    return {
        "service": "JAGeocoder Backend for Tokyo AI Taxi",
        "version": "3.0.0",
        "status": "running",
        "endpoints": [
            "/geocode/{address} - Convert address to coordinates",
            "/distance - Calculate distance between points",
            "/health - Health check",
            "/test/{query} - Test geocoding functionality"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
