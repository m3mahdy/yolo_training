"""
Generate Comprehensive BDD100K Dataset Exploration PDF Report

This script generates a professional PDF report with all visualizations and statistics
from the BDD100K dataset exploration, including class distributions, attribute analysis,
dataset comparisons, and visual samples.

Usage:
    python generate_exploration_report.py
"""

import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors


def load_metadata(metadata_path: Path) -> dict:
    """Load metadata JSON file."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract basic info
    stats = {
        'dataset': data.get('dataset', 'Unknown'),
        'split': data.get('split', 'Unknown'),
        'total_images': data.get('total_files', 0),
        'created': data.get('created', 'Unknown'),
    }
    
    # Calculate aggregate statistics from files
    files = data.get('files', {})
    class_counts = {}
    weather_counts = {}
    scene_counts = {}
    timeofday_counts = {}
    total_objects = 0
    
    for file_data in files.values():
        # Count objects by class
        for cls, count in file_data.get('class_counts', {}).items():
            class_counts[cls] = class_counts.get(cls, 0) + count
        
        # Count weather conditions
        weather = file_data.get('weather', 'unknown')
        weather_counts[weather] = weather_counts.get(weather, 0) + 1
        
        # Count scenes
        scene = file_data.get('scene', 'unknown')
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        
        # Count time of day
        timeofday = file_data.get('timeofday', 'unknown')
        timeofday_counts[timeofday] = timeofday_counts.get(timeofday, 0) + 1
        
        # Total objects
        total_objects += file_data.get('object_count', 0)
    
    stats['total_objects'] = total_objects
    stats['class_distribution'] = class_counts
    stats['weather_distribution'] = weather_counts
    stats['scene_distribution'] = scene_counts
    stats['timeofday_distribution'] = timeofday_counts
    stats['num_classes'] = len(class_counts)
    
    return stats


def create_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    
    return {
        'title': title_style,
        'heading1': heading1_style,
        'heading2': heading2_style,
        'body': body_style,
        'caption': caption_style,
        'date': date_style
    }


def add_title_page(story, styles, full_stats, limited_stats):
    """Add title page with executive summary."""
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("BDD100K Dataset Exploration Report", styles['title']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['heading1']))
    
    # Format numbers safely
    full_images = full_stats.get('total_images', 0)
    limited_images = limited_stats.get('total_images', 0)
    full_images_str = f"{full_images:,}" if isinstance(full_images, int) else str(full_images)
    limited_images_str = f"{limited_images:,}" if isinstance(limited_images, int) else str(limited_images)
    
    summary_text = f"""
    This report provides a comprehensive analysis of the BDD100K dataset prepared for YOLO object detection training. 
    The dataset consists of two versions: a full dataset (~100k images) and a limited representative dataset. 
    The analysis includes class distribution, attribute analysis (weather, scene, time of day), dataset comparisons, 
    and visual samples with annotations.
    <br/><br/>
    <b>Key Statistics:</b><br/>
    • Full Dataset: {full_images_str} images across train/val/test splits<br/>
    • Limited Dataset: {limited_images_str} representative images<br/>
    • Classes: {full_stats.get('num_classes', 0)} object categories<br/>
    • Attributes: Weather conditions, scene types, time of day<br/>
    • Quality: Complete integrity checks performed
    """
    story.append(Paragraph(summary_text, styles['body']))
    story.append(PageBreak())


def add_dataset_overview(story, styles, full_stats, limited_stats):
    """Add dataset overview section with statistics table."""
    story.append(Paragraph("1. Dataset Overview", styles['heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    # Safely format statistics
    def format_stat(value, default=0):
        """Format statistics safely handling various types."""
        if isinstance(value, (int, float)):
            return f"{int(value):,}"
        return str(default)
    
    # Dataset Statistics Table
    dataset_table_data = [
        ['Metric', 'Full Dataset', 'Limited Dataset'],
        ['Total Images', 
         format_stat(full_stats.get('total_images')), 
         format_stat(limited_stats.get('total_images'))],
        ['Train Images', 
         format_stat(full_stats.get('train', {}).get('num_images')),
         format_stat(limited_stats.get('train', {}).get('num_images'))],
        ['Val Images', 
         format_stat(full_stats.get('val', {}).get('num_images')),
         format_stat(limited_stats.get('val', {}).get('num_images'))],
        ['Test Images', 
         format_stat(full_stats.get('test', {}).get('num_images')),
         format_stat(limited_stats.get('test', {}).get('num_images'))],
        ['Number of Classes', 
         str(full_stats.get('num_classes', 0)),
         str(limited_stats.get('num_classes', 0))],
    ]
    
    dataset_table = Table(dataset_table_data, colWidths=[2.5*inch, 2*inch, 2*inch])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    
    story.append(dataset_table)
    story.append(Spacer(1, 0.3*inch))
    
    overview_text = """
    The BDD100K dataset is one of the largest and most diverse driving datasets, featuring images captured 
    across various geographic locations, weather conditions, times of day, and scene types. The full dataset 
    provides comprehensive training data, while the limited dataset offers a representative subset for 
    faster experimentation and validation.
    """
    story.append(Paragraph(overview_text, styles['body']))
    story.append(PageBreak())


def add_class_distribution(story, styles, report_dir):
    """Add class distribution analysis section."""
    story.append(Paragraph("2. Class Distribution Analysis", styles['heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    # Full Dataset Class Distribution
    story.append(Paragraph("2.1 Class Distribution Comparison", styles['heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    class_dist_full_path = report_dir / 'class_distribution_comparison.png'
    if class_dist_full_path.exists():
        img = Image(str(class_dist_full_path), width=7.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Figure 2.1: Class distribution comparison across train, validation, and test splits. "
            "The chart shows the count of objects per class for each split, demonstrating the class imbalance inherent in real-world driving scenarios.",
            styles['caption']
        ))
    else:
        story.append(Paragraph("⚠️ Class distribution chart not found. Please run the analysis notebook first.", styles['body']))
    
    story.append(PageBreak())
    
    # Side-by-side Class Distribution
    story.append(Paragraph("2.2 Side-by-Side Class Distribution Comparison", styles['heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    class_dist_limited_path = report_dir / 'class_distribution_sidebyside.png'
    if class_dist_limited_path.exists():
        img = Image(str(class_dist_limited_path), width=7*inch, height=5*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Figure 2.2: Side-by-side comparison of class distributions showing train, validation, and test splits together. "
            "This view highlights the consistency of class distribution across different dataset splits.",
            styles['caption']
        ))
    else:
        story.append(Paragraph("⚠️ Limited dataset chart not found.", styles['body']))
    
    story.append(PageBreak())


def add_attribute_analysis(story, styles, report_dir):
    """Add attribute analysis section."""
    story.append(Paragraph("3. Attribute Analysis", styles['heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    # Weather Distribution
    story.append(Paragraph("3.1 Weather Conditions Distribution", styles['heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    weather_full_path = report_dir / 'attribute_distribution_weather.png'
    if weather_full_path.exists():
        img = Image(str(weather_full_path), width=7*inch, height=4.2*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Figure 3.1: Distribution of weather conditions across the dataset. The variety of weather conditions "
            "enables robust model training for diverse driving scenarios.",
            styles['caption']
        ))
    else:
        story.append(Paragraph("⚠️ Weather distribution chart not found.", styles['body']))
    
    story.append(PageBreak())
    
    # Scene Distribution
    story.append(Paragraph("3.2 Scene Types Distribution", styles['heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    scene_full_path = report_dir / 'attribute_distribution_scene.png'
    if scene_full_path.exists():
        img = Image(str(scene_full_path), width=7*inch, height=4.2*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Figure 3.2: Distribution of scene types (city street, highway, residential, parking lot, gas station, tunnel). "
            "This diversity ensures the model can generalize across different driving environments.",
            styles['caption']
        ))
    else:
        story.append(Paragraph("⚠️ Scene distribution chart not found.", styles['body']))
    
    story.append(PageBreak())
    
    # Time of Day Distribution
    story.append(Paragraph("3.3 Time of Day Distribution", styles['heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    time_full_path = report_dir / 'attribute_distribution_timeofday.png'
    if time_full_path.exists():
        img = Image(str(time_full_path), width=7*inch, height=3.7*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Figure 3.3: Distribution of images by time of day (daytime, dawn/dusk, night). This temporal diversity "
            "helps the model perform well under varying lighting conditions.",
            styles['caption']
        ))
    else:
        story.append(Paragraph("⚠️ Time of day distribution chart not found.", styles['body']))
    
    story.append(PageBreak())


def add_dataset_comparison(story, styles, report_dir):
    """Add dataset comparison section."""
    story.append(Paragraph("4. Full vs Limited Dataset Comparison", styles['heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    comparison_text = """
    The limited dataset is carefully curated to maintain representative samples across all dimensions while 
    reducing total size. This enables faster iteration during model development while ensuring comprehensive 
    coverage of all classes, weather conditions, scene types, and times of day.
    """
    story.append(Paragraph(comparison_text, styles['body']))
    story.append(Spacer(1, 0.2*inch))
    
    comparison_path = report_dir / 'dataset_comparison_full_vs_limited.png'
    if comparison_path.exists():
        img = Image(str(comparison_path), width=7*inch, height=2.6*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Figure 4.1: Side-by-side comparison of class distributions between full and limited datasets. "
            "Percentages show the representation of each class relative to the full dataset.",
            styles['caption']
        ))
    else:
        story.append(Paragraph("⚠️ Dataset comparison chart not found.", styles['body']))
    
    story.append(PageBreak())


def add_visual_samples(story, styles, report_dir):
    """Add visual samples section."""
    story.append(Paragraph("5. Visual Samples with Annotations", styles['heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    samples_text = """
    The following visualizations demonstrate the annotation quality and diversity of the dataset. 
    Representative samples show single-class annotations for clarity, while complex scenes display 
    multi-object annotations to illustrate real-world driving scenarios.
    """
    story.append(Paragraph(samples_text, styles['body']))
    story.append(Spacer(1, 0.2*inch))
    
    # Add class samples (per-class visualization)
    story.append(Paragraph("5.1 Per-Class Representative Samples", styles['heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    class_samples = sorted(report_dir.glob('class_samples_*.png'))
    if class_samples:
        for i, sample_path in enumerate(class_samples[:6], 1):  # Limit to 6 class samples
            class_name = sample_path.stem.replace('class_samples_', '').replace('_', ' ').title()
            img = Image(str(sample_path), width=7*inch, height=4.6*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(
                f"Figure 5.{i}: Representative samples for {class_name} class with bounding box annotations.",
                styles['caption']
            ))
            story.append(Spacer(1, 0.3*inch))
    
    story.append(PageBreak())
    
    # Add complex scenes (multi-object scenes)
    story.append(Paragraph("5.2 Complex Multi-Object Scenes", styles['heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    complex_scenes = sorted(report_dir.glob('complex_scene_*.png'))
    if complex_scenes:
        for i, scene_path in enumerate(complex_scenes[:4], 1):  # Limit to 4 complex scenes
            img = Image(str(scene_path), width=7*inch, height=4.6*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(
                f"Figure 5.{6+i}: Complex driving scene with multiple object classes annotated simultaneously.",
                styles['caption']
            ))
            story.append(Spacer(1, 0.2*inch))
    
    if not class_samples and not complex_scenes:
        story.append(Paragraph("⚠️ Sample images not found.", styles['body']))
    
    story.append(PageBreak())


def add_conclusions(story, styles, full_stats, limited_stats):
    """Add conclusions and recommendations section."""
    story.append(Paragraph("6. Conclusions and Recommendations", styles['heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    # Calculate percentage safely
    full_total = full_stats.get('total_images', 1)
    limited_total = limited_stats.get('total_images', 0)
    
    if isinstance(full_total, int) and isinstance(limited_total, int) and full_total > 0:
        limited_pct = (limited_total / full_total * 100)
    else:
        limited_pct = 0.0
    
    conclusions_text = f"""
    <b>Dataset Readiness:</b><br/>
    The BDD100K dataset has been successfully prepared and analyzed for YOLO object detection training. 
    Both full and limited versions are ready for immediate use.
    <br/><br/>
    <b>Key Findings:</b><br/>
    • The dataset exhibits natural class imbalance typical of driving scenarios<br/>
    • Comprehensive coverage across weather conditions, scenes, and times of day<br/>
    • High-quality annotations verified through integrity checks<br/>
    • Limited dataset maintains representative coverage (~{limited_pct:.1f}% of full dataset)

    <br/><br/>
    <b>Next Steps:</b><br/>
    • Proceed with YOLO model training using prepared datasets<br/>
    """
    story.append(Paragraph(conclusions_text, styles['body']))


def main():
    """Main function to generate the PDF report."""
    print("=" * 90)
    print("GENERATING COMPREHENSIVE BDD100K DATASET EXPLORATION REPORT")
    print("=" * 90)
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    REPORT_DIR = BASE_DIR / 'data_explore'
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("\n📊 Loading metadata files...")
    try:
        # Find dataset directories
        full_dataset_dir = BASE_DIR / 'bdd100k_yolo'
        limited_dataset_dir = BASE_DIR / 'bdd100k_yolo_limited'
        
        if not full_dataset_dir.exists():
            full_dataset_dir = BASE_DIR / 'bdd100k_yolo_x'
        if not limited_dataset_dir.exists():
            for alt_dir in ['bdd100k_yolo_limited_x', 'bdd100k_yolo_tiny']:
                alt_path = BASE_DIR / alt_dir
                if alt_path.exists():
                    limited_dataset_dir = alt_path
                    break
        
        if not full_dataset_dir.exists():
            raise FileNotFoundError("Full dataset directory not found")
        if not limited_dataset_dir.exists():
            raise FileNotFoundError("Limited dataset directory not found")
        
        # Load all splits for full dataset
        full_stats = {'train': {}, 'val': {}, 'test': {}, 'total_images': 0, 'total_objects': 0}
        splits = ['train', 'val', 'test']
        all_classes = set()
        
        for split in splits:
            metadata_path = full_dataset_dir / 'representative_json' / f'{split}_metadata.json'
            if metadata_path.exists():
                split_data = load_metadata(metadata_path)
                full_stats[split] = {
                    'num_images': split_data.get('total_images', 0),
                    'num_objects': split_data.get('total_objects', 0)
                }
                full_stats['total_images'] += split_data.get('total_images', 0)
                full_stats['total_objects'] += split_data.get('total_objects', 0)
                all_classes.update(split_data.get('class_distribution', {}).keys())
        
        full_stats['class_names'] = sorted(list(all_classes))
        full_stats['num_classes'] = len(all_classes)
        
        # Load all splits for limited dataset
        limited_stats = {'train': {}, 'val': {}, 'test': {}, 'total_images': 0, 'total_objects': 0}
        all_classes_limited = set()
        
        for split in splits:
            metadata_path = limited_dataset_dir / 'representative_json' / f'{split}_metadata.json'
            if metadata_path.exists():
                split_data = load_metadata(metadata_path)
                limited_stats[split] = {
                    'num_images': split_data.get('total_images', 0),
                    'num_objects': split_data.get('total_objects', 0)
                }
                limited_stats['total_images'] += split_data.get('total_images', 0)
                limited_stats['total_objects'] += split_data.get('total_objects', 0)
                all_classes_limited.update(split_data.get('class_distribution', {}).keys())
        
        limited_stats['class_names'] = sorted(list(all_classes_limited))
        limited_stats['num_classes'] = len(all_classes_limited)
        
        # Display loaded values
        print(f"✓ Loaded full dataset metadata from: {full_dataset_dir.name}")
        print(f"  Total images: {full_stats['total_images']:,}")
        print(f"  Train: {full_stats['train'].get('num_images', 0):,}, Val: {full_stats['val'].get('num_images', 0):,}, Test: {full_stats['test'].get('num_images', 0):,}")
        print(f"  Classes: {full_stats['num_classes']}")
        print(f"✓ Loaded limited dataset metadata from: {limited_dataset_dir.name}")
        print(f"  Total images: {limited_stats['total_images']:,}")
        print(f"  Train: {limited_stats['train'].get('num_images', 0):,}, Val: {limited_stats['val'].get('num_images', 0):,}, Test: {limited_stats['test'].get('num_images', 0):,}")
        print(f"  Classes: {limited_stats['num_classes']}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the metadata files exist in the dataset directories.")
        return
    
    # Report file path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'bdd100k_exploration_report_{timestamp}.pdf'
    report_path = REPORT_DIR / report_filename
    
    print(f"\n📄 Creating PDF report: {report_path}")
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Create styles
    styles = create_styles()
    
    # Build story
    story = []
    
    print("\n📝 Building report sections...")
    add_title_page(story, styles, full_stats, limited_stats)
    print("  ✓ Title page and executive summary")
    
    add_dataset_overview(story, styles, full_stats, limited_stats)
    print("  ✓ Dataset overview")
    
    add_class_distribution(story, styles, REPORT_DIR)
    print("  ✓ Class distribution analysis")
    
    add_attribute_analysis(story, styles, REPORT_DIR)
    print("  ✓ Attribute analysis")
    
    add_dataset_comparison(story, styles, REPORT_DIR)
    print("  ✓ Dataset comparison")
    
    add_visual_samples(story, styles, REPORT_DIR)
    print("  ✓ Visual samples")
    
    add_conclusions(story, styles, full_stats, limited_stats)
    print("  ✓ Conclusions and recommendations")
    
    # Build PDF
    print("\n📦 Building PDF document...")
    try:
        doc.build(story)
        print(f"\n✅ Report generated successfully!")
        print(f"📄 Report saved to: {report_path}")
        print(f"📊 File size: {report_path.stat().st_size / (1024*1024):.2f} MB")
        print("\n" + "=" * 90)
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 90)


if __name__ == '__main__':
    main()
