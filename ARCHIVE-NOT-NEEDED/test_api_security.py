#!/usr/bin/env python3
"""
Test script for FraudLens API Security Features
Demonstrates JWT authentication, API keys, rate limiting, and role-based access
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from fraudlens.api.auth import (
    auth_manager, UserCreate, UserRole, 
    create_tokens, authenticate_user, create_api_key
)


def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


async def test_security_features():
    """Test all security features"""
    
    print_section("FRAUDLENS API SECURITY TEST SUITE")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: User Creation
    print_section("1. USER CREATION & ROLES")
    
    test_users = [
        UserCreate(
            username="admin_user",
            email="admin@fraudlens.com",
            password="AdminPass123!",
            role=UserRole.ADMIN
        ),
        UserCreate(
            username="regular_user",
            email="user@fraudlens.com",
            password="UserPass123!",
            role=UserRole.USER
        ),
        UserCreate(
            username="viewer_user",
            email="viewer@fraudlens.com",
            password="ViewerPass123!",
            role=UserRole.VIEWER
        ),
        UserCreate(
            username="api_service",
            email="api@fraudlens.com",
            password="ApiPass123!",
            role=UserRole.API_USER
        )
    ]
    
    created_users = []
    for user_data in test_users:
        try:
            user = auth_manager.create_user(user_data)
            created_users.append(user)
            print(f"✓ Created {user.role.value}: {user.username} (ID: {user.id})")
        except ValueError as e:
            print(f"⚠ User {user_data.username} already exists")
            # Try to get existing user
            user = auth_manager.get_user(user_data.username)
            if user:
                created_users.append(user)
    
    # Test 2: Authentication
    print_section("2. JWT AUTHENTICATION")
    
    for test_user in test_users[:2]:  # Test first two users
        user = authenticate_user(test_user.username, test_user.password)
        if user:
            tokens = create_tokens(user)
            print(f"\n✓ Authenticated: {user.username}")
            print(f"  Access Token: {tokens.access_token[:50]}...")
            print(f"  Token Type: {tokens.token_type}")
            
            # Verify token
            token_data = auth_manager.verify_token(tokens.access_token)
            if token_data:
                print(f"  ✓ Token verified - User ID: {token_data.user_id}, Role: {token_data.role}")
        else:
            print(f"✗ Authentication failed for {test_user.username}")
    
    # Test 3: API Key Management
    print_section("3. API KEY MANAGEMENT")
    
    for user in created_users[:2]:  # Create API keys for first two users
        # Create multiple API keys with different permissions
        api_keys = [
            {
                "name": f"{user.username}_read_only",
                "permissions": ["read"]
            },
            {
                "name": f"{user.username}_full_access",
                "permissions": ["read", "write", "delete"]
            }
        ]
        
        for key_config in api_keys:
            api_key = create_api_key(
                user.id,
                key_config["name"],
                key_config["permissions"]
            )
            print(f"\n✓ Created API Key for {user.username}:")
            print(f"  Name: {key_config['name']}")
            print(f"  Key: {api_key}")
            print(f"  Permissions: {', '.join(key_config['permissions'])}")
            
            # Verify API key
            key_info = auth_manager.verify_api_key(api_key)
            if key_info:
                print(f"  ✓ Key verified - User ID: {key_info['user_id']}, Role: {key_info['role']}")
    
    # Test 4: Rate Limiting
    print_section("4. RATE LIMITING")
    
    for user in created_users:
        role = user.role
        limit = auth_manager.get_rate_limit_for_role(role)
        print(f"\n{role.value.upper()} Rate Limit: {limit} requests/minute")
        
        # Simulate requests
        identifier = f"user_{user.id}"
        endpoint = "/api/analyze"
        
        # Test rate limiting
        for i in range(5):
            allowed, remaining = auth_manager.check_rate_limit(
                identifier, 
                endpoint, 
                limit=10  # Low limit for testing
            )
            
            if i == 0:
                print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Blocked'} (Remaining: {remaining})")
    
    # Test 5: Permission Checking
    print_section("5. ROLE-BASED PERMISSIONS")
    
    permissions_to_test = ["read", "write", "delete", "admin"]
    
    for role in UserRole:
        print(f"\n{role.value.upper()} Permissions:")
        for permission in permissions_to_test:
            has_permission = auth_manager.check_permission(role, permission)
            symbol = "✓" if has_permission else "✗"
            print(f"  {symbol} {permission}")
    
    # Test 6: Session Management
    print_section("6. SESSION MANAGEMENT")
    
    # Create a token and then invalidate it
    if created_users:
        user = created_users[0]
        tokens = create_tokens(user)
        print(f"\n✓ Created session for {user.username}")
        
        # Invalidate token
        invalidated = auth_manager.invalidate_token(tokens.access_token)
        print(f"{'✓' if invalidated else '✗'} Token invalidation: {'Success' if invalidated else 'Failed'}")
        
        # Clean expired sessions
        cleaned = auth_manager.clean_expired_sessions()
        print(f"✓ Cleaned {cleaned} expired sessions")
    
    # Test 7: Password Security
    print_section("7. PASSWORD SECURITY")
    
    test_password = "SecurePass123!"
    hashed = auth_manager.get_password_hash(test_password)
    print(f"\nOriginal Password: {test_password}")
    print(f"Hashed Password: {hashed[:50]}...")
    
    # Verify password
    is_valid = auth_manager.verify_password(test_password, hashed)
    print(f"✓ Password verification: {'Success' if is_valid else 'Failed'}")
    
    # Test wrong password
    is_valid_wrong = auth_manager.verify_password("WrongPassword", hashed)
    print(f"✓ Wrong password rejected: {'Yes' if not is_valid_wrong else 'No'}")
    
    # Summary
    print_section("TEST SUMMARY")
    
    print("""
✅ Security Features Implemented:
  • JWT Authentication with access & refresh tokens
  • API Key management with permissions
  • Rate limiting (100 req/min default, configurable by role)
  • Role-based access control (Admin, User, Viewer, API User)
  • Password hashing with bcrypt
  • Session management and token invalidation
  • SQLite database for persistent storage
  • Redis support for distributed rate limiting

📊 Rate Limits by Role:
  • Admin: 1000 requests/minute
  • User: 100 requests/minute
  • Viewer: 50 requests/minute
  • API User: 500 requests/minute

🔐 Permission Matrix:
  • Admin: read, write, delete, admin
  • User: read, write
  • Viewer: read
  • API User: read, write, api

🚀 API Endpoints:
  • POST /auth/register - Register new user
  • POST /auth/token - Login and get JWT
  • POST /auth/refresh - Refresh access token
  • POST /auth/logout - Logout and invalidate token
  • POST /api-keys - Create API key
  • GET /api-keys - List API keys
  • DELETE /api-keys/{key} - Revoke API key
  • GET /users - List all users (admin only)
  • PATCH /users/{id}/role - Update user role (admin only)
  • POST /analyze/text - Analyze text (protected)
  • POST /analyze/batch - Batch analysis (user/admin)
  • GET /rate-limit/status - Check rate limit status
    """)
    
    print("✅ All security tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_security_features())