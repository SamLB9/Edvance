"""
Authentication configuration for Edvance Study Coach
Simple user management with hashed passwords
"""

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from pathlib import Path

# Default credentials for initial setup
DEFAULT_CREDENTIALS = {
    'usernames': {
        'admin': {
            'name': 'Administrator',
            'email': 'admin@edvance.com',
            'password': '$2b$12$6nUfD.muRyNs3HGCx9SySOtQroa1E0S9X7J0vYyiEPn7.ZNqKIG4G'  # 'password'
        },
        'student': {
            'name': 'Student User',
            'email': 'student@edvance.com',
            'password': '$2b$12$6nUfD.muRyNs3HGCx9SySOtQroa1E0S9X7J0vYyiEPn7.ZNqKIG4G'  # 'password'
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'edvance_study_coach_key',
        'name': 'edvance_cookie'
    }
}

CONFIG_FILE = Path("auth_config.yaml")

# Cache the authenticator instance to avoid duplicate key errors
_authenticator_cache = None
_config_cache = None

def get_authenticator():
    """Get or create the authenticator instance"""
    global _authenticator_cache, _config_cache
    
    # Return cached instances if available
    if _authenticator_cache is not None and _config_cache is not None:
        return _authenticator_cache, _config_cache
    
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as file:
            config = yaml.load(file, Loader=SafeLoader)
    else:
        # Create default config file
        config = DEFAULT_CREDENTIALS
        save_auth_config(config)
    
    # Ensure the config has the correct structure
    if 'credentials' not in config:
        # Convert old format to new format
        config = {
            'credentials': config,
            'cookie': config.get('cookie', DEFAULT_CREDENTIALS['cookie'])
        }
        save_auth_config(config)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    
    # Cache the instances
    _authenticator_cache = authenticator
    _config_cache = config
    
    return authenticator, config

def save_auth_config(config):
    """Save authentication configuration to file"""
    global _authenticator_cache, _config_cache
    
    with open(CONFIG_FILE, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    # Clear cache when config is updated
    _authenticator_cache = None
    _config_cache = None

def hash_password(password):
    """Hash a password for storage"""
    return stauth.Hasher.hash(password)

def add_user(username, email, name, password, config, school_level=None, courses=None, study_goals=None, dob=None, age=None, subject=None):
    """Add a new user to the configuration with extended profile information"""
    hashed_password = hash_password(password)
    user_data = {
        'name': name,
        'email': email,
        'password': hashed_password
    }
    
    # Add extended profile information if provided
    if school_level:
        user_data['school_level'] = school_level
    if courses:
        user_data['courses'] = courses
    if study_goals:
        user_data['study_goals'] = study_goals
    if dob:
        user_data['date_of_birth'] = dob.isoformat()  # Store as ISO format string
    if age:
        user_data['age'] = age
    if subject:
        user_data['subject'] = subject
    
    config['credentials']['usernames'][username] = user_data
    save_auth_config(config)

def remove_user(username, config):
    """Remove a user from the configuration"""
    if username in config['credentials']['usernames']:
        del config['credentials']['usernames'][username]
        save_auth_config(config)

def get_user_info(username, config):
    """Get user information"""
    return config['credentials']['usernames'].get(username, {})

def save_user_course_assignment(username, filename, course, config):
    """Save course assignment for a specific user and file"""
    if username not in config['credentials']['usernames']:
        return False
    
    if 'document_courses' not in config['credentials']['usernames'][username]:
        config['credentials']['usernames'][username]['document_courses'] = {}
    
    config['credentials']['usernames'][username]['document_courses'][filename] = course
    save_auth_config(config)
    return True

def get_user_course_assignments(username, config):
    """Get all course assignments for a user"""
    user_data = config['credentials']['usernames'].get(username, {})
    return user_data.get('document_courses', {})

def remove_user_course_assignment(username, filename, config):
    """Remove course assignment for a specific user and file"""
    if username not in config['credentials']['usernames']:
        return False
    
    user_data = config['credentials']['usernames'][username]
    if 'document_courses' in user_data and filename in user_data['document_courses']:
        del user_data['document_courses'][filename]
        save_auth_config(config)
        return True
    return False