#!/usr/bin/env python3
import re

# Read the app.py file
with open('app.py', 'r') as f:
    content = f.read()

# Define the new login logic with case-insensitive username matching
new_login_logic = '''                        # Check credentials with case-insensitive username matching
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
                                st.error('Username/password is incorrect')'''

# Find and replace the old login logic
old_pattern = r'                        # Check credentials\s+users = config\[\'credentials\'\]\[\'usernames\'\]\s+if username_input in users:.*?st\.rerun\(\)'

# Replace with new logic
updated_content = re.sub(old_pattern, new_login_logic, content, flags=re.DOTALL)

# Write the updated content back
with open('app.py', 'w') as f:
    f.write(updated_content)

print("✅ Updated login logic with case-insensitive username matching")
