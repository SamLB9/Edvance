#!/usr/bin/env python3
import re

# Read the app.py file
with open('app.py', 'r') as f:
    content = f.read()

# Update the tab selection to be dynamic
new_tab_selection = '''        # Tab selection with dynamic switching
        # Check if we should show login tab after successful signup
        default_tab = 0  # Default to login tab
        if st.session_state.get('show_login_tab', False):
            default_tab = 0  # Show login tab
            st.session_state['show_login_tab'] = False  # Reset the flag
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])'''

# Update the success message to use toast and redirect
new_success_handling = '''                                # Add user to config with extended profile information
                                add_user(username_signup, email, full_name, password_signup, config, 
                                        school_level, courses, study_goals)
                                
                                # Show success toast and redirect to login tab
                                st.toast("✅ Account created successfully! You can now login.", icon="🎉", duration=5)
                                st.session_state['show_login_tab'] = True  # Flag to show login tab
                                st.rerun()'''

# Replace the tab selection
content = re.sub(
    r'        # Tab selection\s+tab1, tab2 = st\.tabs\(\["🔐 Login", "📝 Sign Up"\]\)',
    new_tab_selection,
    content
)

# Replace the success handling
content = re.sub(
    r'                                st\.success\("Account created successfully! You can now login\."\)\s+st\.balloons\(\)',
    new_success_handling,
    content
)

# Write the updated content back
with open('app.py', 'w') as f:
    f.write(content)

print("✅ Updated signup to redirect to login tab with toast message")
