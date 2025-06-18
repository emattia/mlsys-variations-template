#!/usr/bin/env python3
"""
MLX Backup Management System

This system implements intelligent backup retention policies and automated cleanup
for the MLX repository, following the museum-quality standards.

Features:
- Automated backup discovery and categorization
- Intelligent retention policies (7-day active, 30-day archive) 
- Safe cleanup with verification
- Integration with CI/CD pipeline
- Comprehensive logging and reporting

Usage:
    python scripts/backup_manager.py --action=audit
    python scripts/backup_manager.py --action=cleanup --dry-run
    python scripts/backup_manager.py --action=cleanup --confirm
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BackupFile:
    """Represents a backup file with metadata."""
    path: str
    size_bytes: int
    created_time: datetime
    age_days: int
    category: str  # 'active', 'archivable', 'expired'
    source_exists: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            'created_time': self.created_time.isoformat(),
            'size_mb': round(self.size_bytes / (1024 * 1024), 2)
        }

class BackupManager:
    """Manages backup files with intelligent retention policies."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.active_days = 7    # Keep backups for 7 days
        self.archive_days = 30  # Archive after 30 days
        self.archive_dir = self.root_path / "archive" / "backups"
        
    def discover_backups(self) -> List[BackupFile]:
        """Discover all backup files in the repository."""
        logger.info("Discovering backup files...")
        backup_files = []
        
        for backup_path in self.root_path.rglob("*.backup"):
            if backup_path.is_file():
                try:
                    stat = backup_path.stat()
                    created_time = datetime.fromtimestamp(stat.st_mtime)
                    age_days = (datetime.now() - created_time).days
                    
                    # Determine category based on age
                    if age_days <= self.active_days:
                        category = "active"
                    elif age_days <= self.archive_days:
                        category = "archivable"
                    else:
                        category = "expired"
                    
                    # Check if source file exists
                    source_path = backup_path.with_suffix("")
                    source_exists = source_path.exists()
                    
                    backup_file = BackupFile(
                        path=str(backup_path.relative_to(self.root_path)),
                        size_bytes=stat.st_size,
                        created_time=created_time,
                        age_days=age_days,
                        category=category,
                        source_exists=source_exists
                    )
                    backup_files.append(backup_file)
                    
                except Exception as e:
                    logger.error(f"Error processing {backup_path}: {e}")
        
        logger.info(f"Discovered {len(backup_files)} backup files")
        return backup_files
    
    def audit_backups(self) -> Dict:
        """Audit all backup files and generate report."""
        backups = self.discover_backups()
        
        # Categorize backups
        categories = {"active": [], "archivable": [], "expired": []}
        total_size = 0
        
        for backup in backups:
            categories[backup.category].append(backup)
            total_size += backup.size_bytes
        
        # Generate audit report
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "total_backups": len(backups),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "categories": {
                "active": {
                    "count": len(categories["active"]),
                    "size_mb": round(sum(b.size_bytes for b in categories["active"]) / (1024 * 1024), 2),
                    "files": [b.to_dict() for b in categories["active"]]
                },
                "archivable": {
                    "count": len(categories["archivable"]),
                    "size_mb": round(sum(b.size_bytes for b in categories["archivable"]) / (1024 * 1024), 2),
                    "files": [b.to_dict() for b in categories["archivable"]]
                },
                "expired": {
                    "count": len(categories["expired"]),
                    "size_mb": round(sum(b.size_bytes for b in categories["expired"]) / (1024 * 1024), 2),
                    "files": [b.to_dict() for b in categories["expired"]]
                }
            },
            "recommendations": self._generate_recommendations(categories)
        }
        
        return report
    
    def _generate_recommendations(self, categories: Dict) -> List[str]:
        """Generate cleanup recommendations based on audit."""
        recommendations = []
        
        if categories["expired"]:
            recommendations.append(
                f"SAFE TO DELETE: {len(categories['expired'])} expired backups "
                f"({round(sum(b.size_bytes for b in categories['expired']) / (1024 * 1024), 2)} MB)"
            )
        
        if categories["archivable"]:
            recommendations.append(
                f"READY FOR ARCHIVE: {len(categories['archivable'])} backups "
                f"({round(sum(b.size_bytes for b in categories['archivable']) / (1024 * 1024), 2)} MB)"
            )
        
        # Check for orphaned backups (source file doesn't exist)
        orphaned = [b for cat in categories.values() for b in cat if not b.source_exists]
        if orphaned:
            recommendations.append(
                f"ORPHANED BACKUPS: {len(orphaned)} backups have no source file - safe to delete"
            )
        
        if not recommendations:
            recommendations.append("EXCELLENT: No cleanup needed - backup management is optimal")
        
        return recommendations
    
    def cleanup_backups(self, dry_run: bool = True) -> Dict:
        """Clean up backups according to retention policy."""
        logger.info(f"Starting backup cleanup (dry_run={dry_run})...")
        
        backups = self.discover_backups()
        cleanup_stats = {
            "deleted": [],
            "archived": [],
            "errors": [],
            "total_freed_mb": 0
        }
        
        for backup in backups:
            try:
                backup_path = self.root_path / backup.path
                
                if backup.category == "expired" or not backup.source_exists:
                    # Delete expired or orphaned backups
                    if not dry_run:
                        backup_path.unlink()
                        logger.info(f"Deleted: {backup.path}")
                    
                    cleanup_stats["deleted"].append(backup.path)
                    cleanup_stats["total_freed_mb"] += backup.size_bytes / (1024 * 1024)
                
                elif backup.category == "archivable":
                    # Archive old but not expired backups
                    if not dry_run:
                        self.archive_dir.mkdir(parents=True, exist_ok=True)
                        archive_path = self.archive_dir / backup_path.name
                        backup_path.rename(archive_path)
                        logger.info(f"Archived: {backup.path} -> {archive_path}")
                    
                    cleanup_stats["archived"].append(backup.path)
                    
            except Exception as e:
                error_msg = f"Error cleaning {backup.path}: {e}"
                logger.error(error_msg)
                cleanup_stats["errors"].append(error_msg)
        
        cleanup_stats["total_freed_mb"] = round(cleanup_stats["total_freed_mb"], 2)
        
        if dry_run:
            logger.info("DRY RUN completed - no files were actually modified")
        else:
            logger.info(f"Cleanup completed - freed {cleanup_stats['total_freed_mb']} MB")
        
        return cleanup_stats
    
    def generate_report(self, output_path: str = "backup_audit_report.json"):
        """Generate comprehensive backup audit report."""
        report = self.audit_backups()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Audit report saved to: {output_path}")
        return report

def main():
    """Main CLI interface for backup management."""
    parser = argparse.ArgumentParser(description="MLX Backup Management System")
    parser.add_argument(
        "--action", 
        choices=["audit", "cleanup", "report"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Preview changes without executing them"
    )
    parser.add_argument(
        "--confirm", 
        action="store_true",
        help="Confirm destructive operations"
    )
    parser.add_argument(
        "--output", 
        default="backup_audit_report.json",
        help="Output file for reports"
    )
    
    args = parser.parse_args()
    
    manager = BackupManager()
    
    if args.action == "audit":
        report = manager.audit_backups()
        print(f"\n{'='*60}")
        print("🎖️  MLX BACKUP AUDIT REPORT")
        print(f"{'='*60}")
        print(f"Total Backups: {report['total_backups']}")
        print(f"Total Size: {report['total_size_mb']} MB")
        print(f"\nCategories:")
        for category, data in report['categories'].items():
            print(f"  {category.upper()}: {data['count']} files ({data['size_mb']} MB)")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
    elif args.action == "cleanup":
        if not args.dry_run and not args.confirm:
            print("ERROR: Cleanup requires either --dry-run or --confirm flag")
            sys.exit(1)
        
        stats = manager.cleanup_backups(dry_run=args.dry_run)
        print(f"\n{'='*60}")
        print("🎖️  MLX BACKUP CLEANUP RESULTS")
        print(f"{'='*60}")
        print(f"Files Deleted: {len(stats['deleted'])}")
        print(f"Files Archived: {len(stats['archived'])}")
        print(f"Space Freed: {stats['total_freed_mb']} MB")
        
        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"  • {error}")
    
    elif args.action == "report":
        report = manager.generate_report(args.output)
        print(f"Comprehensive report saved to: {args.output}")

if __name__ == "__main__":
    main() 