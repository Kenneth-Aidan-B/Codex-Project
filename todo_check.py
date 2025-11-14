#!/usr/bin/env python3
"""
Todo checker script - verifies completion of all tasks in todo.json
Exits with non-zero status if any required tasks are pending or failed.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('todo.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

TODO_FILE = Path(__file__).parent / "todo.json"
BASE_DIR = Path(__file__).parent


def load_todos() -> Dict:
    """Load the todo.json file"""
    try:
        with open(TODO_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {TODO_FILE}: {e}")
        sys.exit(1)


def save_todos(todos: Dict) -> None:
    """Save the updated todo.json file"""
    try:
        with open(TODO_FILE, 'w') as f:
            json.dump(todos, f, indent=2)
        logger.info(f"Updated {TODO_FILE}")
    except Exception as e:
        logger.error(f"Failed to save {TODO_FILE}: {e}")


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    path = BASE_DIR / filepath
    exists = path.exists()
    if exists:
        logger.info(f"✓ File exists: {filepath}")
    else:
        logger.warning(f"✗ File missing: {filepath}")
    return exists


def check_verification_file(filepath: str) -> bool:
    """Check if a verification marker file exists"""
    path = BASE_DIR / filepath
    exists = path.exists()
    if exists:
        logger.info(f"✓ Verification passed: {filepath}")
    else:
        logger.warning(f"✗ Verification missing: {filepath}")
    return exists


def verify_task(task: Dict) -> bool:
    """Verify a single task completion"""
    task_id = task.get('id', 'unknown')
    task_name = task.get('name', 'unknown')
    
    logger.info(f"\nChecking task {task_id}: {task_name}")
    
    # Check if task file exists
    if 'file' in task:
        if not check_file_exists(task['file']):
            return False
    
    # Check verification file if specified
    if 'verification' in task:
        if not check_verification_file(task['verification']):
            return False
    
    return True


def update_task_status(task: Dict, success: bool) -> None:
    """Update task status based on verification result"""
    if success and task['status'] != 'done':
        task['status'] = 'done'
        task['timestamp'] = datetime.now().isoformat()
        logger.info(f"✓ Task {task['id']} marked as DONE")
    elif not success and task['status'] == 'done':
        task['status'] = 'pending'
        task['timestamp'] = None
        logger.warning(f"✗ Task {task['id']} reverted to PENDING")


def main():
    """Main verification function"""
    logger.info("="*60)
    logger.info("Starting TODO verification")
    logger.info("="*60)
    
    todos = load_todos()
    tasks = todos.get('tasks', [])
    
    if not tasks:
        logger.error("No tasks found in todo.json")
        sys.exit(1)
    
    total_tasks = len(tasks)
    completed_tasks = 0
    pending_tasks = 0
    failed_tasks = 0
    
    # Verify each task
    for task in tasks:
        try:
            verified = verify_task(task)
            update_task_status(task, verified)
            
            if task['status'] == 'done':
                completed_tasks += 1
            else:
                pending_tasks += 1
        except Exception as e:
            logger.error(f"Error verifying task {task.get('id')}: {e}")
            failed_tasks += 1
            task['status'] = 'failed'
    
    # Save updated todos
    save_todos(todos)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Completed: {completed_tasks}")
    logger.info(f"Pending: {pending_tasks}")
    logger.info(f"Failed: {failed_tasks}")
    logger.info("="*60)
    
    # Exit with appropriate code
    if pending_tasks > 0 or failed_tasks > 0:
        logger.warning(f"\n⚠ {pending_tasks + failed_tasks} tasks remaining")
        sys.exit(1)
    else:
        logger.info("\n✓ All tasks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
