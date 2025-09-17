#!/usr/bin/env python3
import re

# Read the app.py file
with open('app.py', 'r') as f:
    content = f.read()

# Define the new login page with tabs
new_login_page = '''def show_login_page(authenticator):
    """Display the login page with sign up functionality"""
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem;">
            <h1>🎓 Edvance Study Coach</h1>
            <p style="font-size: 1.2rem; color: #666;">Your AI-powered study companion</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for Login and Sign Up
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Tab selection
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("### Login to your account")
            
            # Custom login form
            with st.form("login_form"):
                username_input = st.text_input("Username", placeholder="Enter your username")
                password_input = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_button = st.form_submit_button("Login", use_container_width=True)
                
                if submit_button:
                    if username_input and password_input:
                        # Get the authenticator and config
                        from auth_config import get_authenticator
                        auth, config = get_authenticator()
                        
                        # Check credentials with case-insensitive username matching
                        users = config['credentials']['usernames']
                        
                        # Find username with case-insensitive matching
                        matched_username = None
                        if username_input in users:
                            matched_username = username_input
                        else:
                            # Case-insensitive search
                            for stored_username in users.keys():
                                if stored_username.lower() == username_input.lower():
                                    matched_username = stored_username
                                    break
                        
                        if matched_username:
                            stored_password = users[matched_username]['password']
                            # Verify password using streamlit_authenticator's Hasher
                            import streamlit_authenticator as stauth
                            if stauth.Hasher.check_pw(password_input, stored_password):
                                # Login successful
                                st.session_state['authentication_status'] = True
                                st.session_state['username'] = matched_username  # Use the matched username
                                st.session_state['name'] = users[matched_username]['name']
                                
                                # Load extended user profile information if available
                                user_data = users[matched_username]
                                if 'school_level' in user_data:
                                    st.session_state['user_school_level'] = user_data['school_level']
                                if 'courses' in user_data:
                                    st.session_state['user_courses'] = user_data['courses']
                                if 'study_goals' in user_data:
                                    st.session_state['user_study_goals'] = user_data['study_goals']
                                
                                st.rerun()
                            else:
                                st.error('Username/password is incorrect')
                        else:
                            st.error('Username/password is incorrect')
                    else:
                        st.warning('Please enter both username and password')
            
            # Show default credentials info
            with st.expander("ℹ️ Default Credentials", expanded=False):
                st.markdown("**For testing purposes:**")
                st.code("""
Username: admin
Password: password

Username: student  
Password: password
                """)
                st.markdown("*Please change these credentials in production!*")
        
        with tab2:
            st.markdown("### Create a new account")
            
            # Sign up form
            with st.form("signup_form"):
                st.markdown("**Personal Information**")
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email address")
                
                st.markdown("**Account Details**")
                username_signup = st.text_input("Username", placeholder="Choose a username")
                password_signup = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                
                st.markdown("**Academic Information**")
                school_level = st.selectbox(
                    "School Level",
                    ["High School", "Undergraduate", "Graduate", "PhD", "Professional Development", "Other"]
                )
                
                courses = st.text_area(
                    "Courses/Subjects", 
                    placeholder="Enter your courses or subjects (one per line or comma-separated)",
                    help="List the courses or subjects you're studying"
                )
                
                st.markdown("**Additional Information**")
                study_goals = st.text_area(
                    "Study Goals", 
                    placeholder="What are your main study goals?",
                    help="Optional: Describe what you want to achieve with this study coach"
                )
                
                signup_button = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_button:
                    # Validate form
                    errors = []
                    
                    if not full_name.strip():
                        errors.append("Full name is required")
                    if not email.strip():
                        errors.append("Email is required")
                    elif "@" not in email:
                        errors.append("Please enter a valid email address")
                    if not username_signup.strip():
                        errors.append("Username is required")
                    if not password_signup:
                        errors.append("Password is required")
                    elif len(password_signup) < 6:
                        errors.append("Password must be at least 6 characters long")
                    if password_signup != confirm_password:
                        errors.append("Passwords do not match")
                    
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        # Check if username already exists
                        from auth_config import get_authenticator, add_user
                        auth, config = get_authenticator()
                        
                        if username_signup in config['credentials']['usernames']:
                            st.error("Username already exists. Please choose a different username.")
                        else:
                            try:
                                # Add user to config with extended profile information
                                add_user(username_signup, email, full_name, password_signup, config, 
                                        school_level, courses, study_goals)
                                
                                st.success("Account created successfully! You can now login.")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"Error creating account: {str(e)}")'''

# Find and replace the show_login_page function
old_pattern = r'def show_login_page\(authenticator\):.*?(?=def main\(\)|$)'

# Replace with new function
updated_content = re.sub(old_pattern, new_login_page, content, flags=re.DOTALL)

# Write the updated content back
with open('app.py', 'w') as f:
    f.write(updated_content)

print("✅ Updated login page with sign up tab functionality")
