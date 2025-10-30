"""
EXIF Metadata Extraction Tool
Extracts camera metadata from images for authenticity verification
"""
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from backend.models import ExifMetadata, ExifResult
from typing import Optional


def extract_gps_info(gps_data: dict) -> Optional[dict]:
    """Parse GPS information from EXIF data"""
    if not gps_data:
        return None
    
    gps_info = {}
    for key, val in gps_data.items():
        tag = GPSTAGS.get(key, key)
        gps_info[tag] = str(val)
    
    return gps_info


def extract_exif(image_path: str) -> ExifResult:
    """
    Extract EXIF metadata from image file
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ExifResult with status and metadata
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()

        if not exif_data:
            return ExifResult(
                status="no_exif",
                metadata=None,
                error=None
            )
        
        # Parse important EXIF tags
        raw_metadata = {}
        parsed_data = {}
        
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            raw_metadata[tag] = str(value)
            
            # Define a list of tags to parsed_data keys
            tag_mapping = [
                "Make",
                "Model",
                "DateTime",
                "DateTimeOriginal",
                "DateTimeDigitized",
                "Software",
                "Orientation",
                "XResolution",
                "YResolution",
                "Flash",
                "FocalLength",
                "ExposureTime",
                "FNumber",
                "ISOSpeedRatings",
            ]

            # Extract key fields for structured model
            if tag in tag_mapping:
                parsed_data[tag_mapping[tag]] = str(value)
            elif tag == "GPSInfo" and isinstance(value, dict):
                parsed_data["GPSInfo"] = extract_gps_info(value)
        
        # Add raw metadata
        parsed_data["raw_metadata"] = raw_metadata
        
        # Create structured metadata
        metadata = ExifMetadata(**parsed_data)

        return ExifResult(
            status="success",
            metadata=metadata,
            error=None
        )
    
    except Exception as e:
        return ExifResult(
            status="error",
            metadata=None,
            error=str(e)
        )


if __name__ == "__main__":
    # Test with sample images
    test_images = [
        "/home/arka/Desktop/Hackathons/ihub/pic.jpg",
        "/home/arka/Desktop/Hackathons/ihub/pic2.jpg"
    ]
    
    for image_path in test_images:
        print(f"\n{'='*60}")
        print(f"Extracting EXIF from: {image_path}")
        print(f"{'='*60}")
        
        result = extract_exif(image_path)
        
        print(f"Status: {result.status}")
        
        if result.status == "success" and result.metadata:
            print("\n📷 Camera Information:")
            if result.metadata.Make:
                print(f"  Make: {result.metadata.Make}")
            if result.metadata.Model:
                print(f"  Model: {result.metadata.Model}")
            
            print("\n📅 Date/Time:")
            if result.metadata.DateTime:
                print(f"  DateTime: {result.metadata.DateTime}")
            if result.metadata.DateTimeOriginal:
                print(f"  Original: {result.metadata.DateTimeOriginal}")
            
            print("\n⚙️  Settings:")
            if result.metadata.FocalLength:
                print(f"  Focal Length: {result.metadata.FocalLength}")
            if result.metadata.FNumber:
                print(f"  Aperture: {result.metadata.FNumber}")
            if result.metadata.ExposureTime:
                print(f"  Exposure: {result.metadata.ExposureTime}")
            if result.metadata.ISOSpeedRatings:
                print(f"  ISO: {result.metadata.ISOSpeedRatings}")
            
            if result.metadata.Software:
                print(f"\n💻 Software: {result.metadata.Software}")
            
            if result.metadata.GPSInfo:
                print(f"\n📍 GPS Data: Available")
                for key, val in result.metadata.GPSInfo.items():
                    print(f"  {key}: {val}")
        
        elif result.status == "no_exif":
            print("ℹ️  No EXIF data found in image")
        
        elif result.status == "error":
            print(f"❌ Error: {result.error}")