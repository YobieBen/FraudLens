# Gmail Connection Setup for FraudLens

## How to Connect Gmail to FraudLens

### Step 1: Enable 2-Step Verification
1. Go to your Google Account: https://myaccount.google.com/
2. Click on "Security" in the left sidebar
3. Under "How you sign in to Google", click on "2-Step Verification"
4. Follow the prompts to enable it

### Step 2: Generate an App Password
1. Go to: https://myaccount.google.com/apppasswords
2. You may need to sign in again
3. In the "Select app" dropdown, choose "Mail" or "Other (Custom name)"
4. If you chose "Other", type "FraudLens"
5. Click "Generate"
6. **Copy the 16-character password** (it looks like: xxxx xxxx xxxx xxxx)
7. Save this password securely - you won't be able to see it again!

### Step 3: Connect in FraudLens
1. Open FraudLens at http://localhost:7863
2. Go to the "📧 Email Scanner" tab
3. Enter your Gmail address (e.g., yourname@gmail.com)
4. Paste the App Password (without spaces)
5. Click "🔗 Connect"

### Important Notes:
- **DO NOT** use your regular Gmail password - it won't work
- The App Password is different from your Gmail password
- You can revoke App Passwords anytime from your Google Account
- Each App Password can only be viewed once when created

### Troubleshooting:
- If connection fails, make sure 2-Step Verification is enabled
- Check that you're using the App Password, not your regular password
- Make sure IMAP is enabled in Gmail settings:
  - Go to Gmail Settings → See all settings → Forwarding and POP/IMAP
  - Enable IMAP access

### Security:
- App Passwords are secure and limited in scope
- They can only access the specific service (email in this case)
- You can revoke them anytime from your Google Account
- FraudLens only reads emails, never modifies or deletes them