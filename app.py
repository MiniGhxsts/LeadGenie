import os
import time
import shutil
import requests
import pandas as pd
import streamlit as st
import zipfile
import io
import PyPDF2
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from google import genai
from email.message import EmailMessage
from supabase import create_client, Client

# --- Configuration & Setup ---
st.set_page_config(page_title="LeadGenie", layout="wide", page_icon="⚡")

load_dotenv()

# Cloud fallback: support both .env (local) and Streamlit secrets (Streamlit Cloud)
PLACES_API_KEY = os.getenv("PLACES_API_KEY")
if not PLACES_API_KEY and "PLACES_API_KEY" in st.secrets:
    PLACES_API_KEY = st.secrets["PLACES_API_KEY"]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Lemon Squeezy Checkout URLs
LEMON_SQUEEZY_STARTER_URL = os.getenv("LEMON_SQUEEZY_STARTER_URL")
if not LEMON_SQUEEZY_STARTER_URL and "LEMON_SQUEEZY_STARTER_URL" in st.secrets:
    LEMON_SQUEEZY_STARTER_URL = st.secrets["LEMON_SQUEEZY_STARTER_URL"]

LEMON_SQUEEZY_PRO_URL = os.getenv("LEMON_SQUEEZY_PRO_URL")
if not LEMON_SQUEEZY_PRO_URL and "LEMON_SQUEEZY_PRO_URL" in st.secrets:
    LEMON_SQUEEZY_PRO_URL = st.secrets["LEMON_SQUEEZY_PRO_URL"]

# Gemini Model constant
GEMINI_MODEL = "gemini-3.5-flash-lite"

if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    ai_client = None
    st.error("System Error: Missing Gemini API Key in environment or secrets.")

DRAFTS_DIR = "Email_Drafts"
if not os.path.exists(DRAFTS_DIR):
    os.makedirs(DRAFTS_DIR, exist_ok=True)

# --- Supabase Initialization ---
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception:
    st.error("Critical Error: Missing or invalid Supabase credentials in .streamlit/secrets.toml")
    st.stop()

# --- License & Verification Engine ---
def verify_lemon_squeezy_license(license_key, user_email=""):
    """
    Validates license keys from Lemon Squeezy, Supabase licenses table, or Admin Master keys.
    Returns: (is_valid: bool, message: str, credits: int, tier: str)
    """
    key = license_key.strip()
    if not key:
        return False, "Please enter your Lemon Squeezy License Key or Order ID.", 0, "free"
    
    # 1. Admin Master & VIP Testing Keys
    master_keys = {
        "GENIE-VIP": (300, "pro"),
        "FOUNDER-2026": (500, "pro"),
        "HUNTER-TEST": (100, "pro"),
        "GENIE100": (100, "pro"),
        "PRO300": (300, "pro")
    }
    if key.upper() in master_keys:
        credits, tier = master_keys[key.upper()]
        return True, "Authorized Master Key Verified!", credits, tier
        
    # 2. Check Supabase `licenses` table if configured
    try:
        lic_res = supabase.table('licenses').select('*').eq('key', key).execute()
        if lic_res.data and len(lic_res.data) > 0:
            lic = lic_res.data[0]
            if lic.get('is_redeemed', False):
                return False, "This license key has already been redeemed by another user.", 0, "free"
            credits = lic.get('credits', 100)
            tier = lic.get('tier', 'pro')
            try:
                supabase.table('licenses').update({'is_redeemed': True, 'redeemed_by': user_email}).eq('key', key).execute()
            except Exception:
                pass
            return True, "Supabase License Key Verified!", credits, tier
    except Exception:
        pass

    # 3. Call official Lemon Squeezy License Activation API
    try:
        res = requests.post(
            "https://api.lemonsqueezy.com/v1/licenses/activate",
            data={"license_key": key, "instance_name": f"LeadGenie ({user_email})"},
            timeout=5
        )
        if res.status_code == 200:
            data = res.json()
            if data.get("activated"):
                prod_name = data.get("meta", {}).get("product_name", "").lower()
                credits = 300 if ("300" in prod_name or "pro" in prod_name) else 100
                return True, "Lemon Squeezy License Successfully Activated!", credits, "pro"
        err_msg = res.json().get("error", "License key not found or activation limit reached.")
        return False, f"License Verification Failed: {err_msg}", 0, "free"
    except Exception as e:
        return False, f"License server verification error: {e}", 0, "free"

# --- Gemini Quota & Health Monitoring ---
@st.cache_data(ttl=120)
def check_gemini_quota_status():
    """
    Actively queries Gemini with a tiny 1-word prompt to detect whether 
    the API key has valid usage credits or has hit 429 RESOURCE_EXHAUSTED.
    """
    if not GEMINI_API_KEY or not ai_client:
        return "MISSING_KEY", "Gemini API key is not configured.", "🔴"
    try:
        test_resp = ai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Say OK."
        )
        if test_resp and test_resp.text:
            return "OPERATIONAL", "Gemini AI is operational and quota is active.", "🟢"
        return "DEGRADED", "Gemini AI responded with empty output.", "🟡"
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
            return "QUOTA_EXHAUSTED", "Google Gemini API has exhausted its usage credits or rate limit (429).", "🔴"
        elif "billing" in err_str or "credit" in err_str:
            return "BILLING_REQUIRED", "Google Cloud billing account requires active credits or payment method.", "🔴"
        elif "api_key" in err_str or "403" in err_str or "permission" in err_str:
            return "INVALID_KEY", "Gemini API key is invalid or unauthorized.", "🔴"
        elif "503" in err_str or "unavailable" in err_str:
            return "HIGH_DEMAND", "Gemini AI is temporarily experiencing high server demand (503).", "🟡"
        return "ERROR", f"Gemini error: {str(e)[:60]}", "🔴"

# --- Authentication Gate (Paid Members Only) ---
if 'user' not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
        else:
            st.title("LeadGenie SaaS")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            auth_mode = st.radio("Access Gate", ["Member Sign In", "Activate Paid License"], horizontal=True)
            st.markdown("---")
            
            if auth_mode == "Member Sign In":
                st.markdown("##### Existing Customer Login")
                st.caption("Sign in with your registered account credentials.")
                email = st.text_input("Account Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_pass")
                
                if st.button("Sign In to LeadGenie", type="primary", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user = res.user
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign In Failed: {e}")
                        
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("Don't have an account yet? Access is strictly paid. Switch to 'Activate Paid License' above with your Lemon Squeezy order key.")
                
            else: # "Activate Paid License"
                st.markdown("##### Activate Your Paid Membership")
                st.markdown("LeadGenie is a members-only application. Enter your Lemon Squeezy purchase License Key or Order ID below to create your specialized login.")
                
                license_key = st.text_input("Lemon Squeezy License Key / Order ID", placeholder="e.g. XXXXX-XXXXX-XXXXX or GENIE-VIP", key="signup_lic")
                email = st.text_input("Your Email (used for receipt)", key="signup_email")
                password = st.text_input("Create Password", type="password", key="signup_pass")
                
                if st.button("Verify License & Create Account", type="primary", use_container_width=True):
                    if not email or not password or not license_key:
                        st.warning("Please fill in all fields (Email, Password, and License Key).")
                    else:
                        is_valid, lic_msg, grant_credits, grant_tier = verify_lemon_squeezy_license(license_key, email)
                        if is_valid:
                            try:
                                res = supabase.auth.sign_up({"email": email, "password": password})
                                if res.user:
                                    st.session_state.user = res.user
                                    # Provision profile with verified paid credits
                                    try:
                                        supabase.table('user_profiles').insert({
                                            'user_id': res.user.id,
                                            'target_location': "Mid-sized accounting firms in Rochester NY",
                                            'target_role': "Audit Intern",
                                            'leads_to_generate': 5,
                                            'credits_remaining': grant_credits,
                                            'tier': grant_tier
                                        }).execute()
                                    except Exception:
                                        pass
                                    st.success(f"🎉 {lic_msg} Your account has been provisioned with {grant_credits} leads!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.info("Account created! Please check your email to confirm registration or sign in.")
                            except Exception as e:
                                st.error(f"Registration Error: {e}")
                        else:
                            st.error(f"❌ {lic_msg}")
                            starter_url = LEMON_SQUEEZY_STARTER_URL or "https://lemonsqueezy.com"
                            st.markdown(f"""
                                <div style="margin-top:12px; padding:12px; background-color:#FEF2F2; border:1px solid #FCA5A5; border-radius:6px; font-size:13px; color:#991B1B;">
                                    <b>Need a License?</b> This web app is private and paid. You must purchase an access pass to receive a license key.<br><br>
                                    <a href="{starter_url}" target="_blank" style="color:#DC2626; font-weight:700; text-decoration:underline;">👉 Purchase Access on Lemon Squeezy ($19)</a>
                                </div>
                            """, unsafe_allow_html=True)
    st.stop()

# --- Session State Initialization (Post-Auth) ---
if 'total_searches' not in st.session_state: st.session_state.total_searches = 0
if 'successful_emails' not in st.session_state: st.session_state.successful_emails = 0
if 'show_success_banner' not in st.session_state: st.session_state.show_success_banner = False
if 'last_processed_count' not in st.session_state: st.session_state.last_processed_count = 0
if 'ai_quota_depleted' not in st.session_state: st.session_state.ai_quota_depleted = False

if 'resume_text' not in st.session_state: st.session_state.resume_text = "" 
if 'resume_filename' not in st.session_state: st.session_state.resume_filename = ""
if 'drafts_memory' not in st.session_state: st.session_state.drafts_memory = {}

if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# Load Cloud Preferences & Credit Quotas safely
if 'config_loaded' not in st.session_state:
    try:
        user_prof = supabase.table('user_profiles').select('*').eq('user_id', st.session_state.user.id).execute()
        if user_prof.data:
            profile = user_prof.data[0]
            st.session_state.input_location = profile.get('target_location', "Mid-sized accounting firms in Rochester NY")
            st.session_state.input_role = profile.get('target_role', "Audit Intern")
            st.session_state.input_num = profile.get('leads_to_generate', 5)
            
            raw_credits = profile.get('credits_remaining')
            st.session_state.credits_remaining = int(raw_credits) if raw_credits is not None else 10
            st.session_state.tier = profile.get('tier', 'free')
        else:
            st.session_state.input_location = "Mid-sized accounting firms in Rochester NY"
            st.session_state.input_role = "Audit Intern"
            st.session_state.input_num = 5
            st.session_state.credits_remaining = 10
            st.session_state.tier = 'free'
            try:
                supabase.table('user_profiles').insert({
                    'user_id': st.session_state.user.id,
                    'target_location': st.session_state.input_location,
                    'target_role': st.session_state.input_role,
                    'leads_to_generate': st.session_state.input_num,
                    'credits_remaining': 10,
                    'tier': 'free'
                }).execute()
            except Exception:
                supabase.table('user_profiles').insert({
                    'user_id': st.session_state.user.id,
                    'target_location': st.session_state.input_location,
                    'target_role': st.session_state.input_role,
                    'leads_to_generate': st.session_state.input_num
                }).execute()
    except Exception:
        st.session_state.input_location = "Mid-sized accounting firms in Rochester NY"
        st.session_state.input_role = "Audit Intern"
        st.session_state.input_num = 5
        st.session_state.credits_remaining = 10
        st.session_state.tier = 'free'
    st.session_state.config_loaded = True

# --- Core Functions ---
def extract_resume_text(uploaded_file):
    if not uploaded_file: return ""
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        return uploaded_file.read().decode("utf-8")

def create_eml_draft(firm_name, hr_name, hr_email, location, target_role, resume_text, user_id=None, status_ui=None):
    if hr_email in ["Not Found", "Error", "API Overload", "N/A"]: return "N/A"

    msg = EmailMessage()
    msg['To'] = hr_email
    msg['From'] = "YOUR_EMAIL@gmail.com"
    msg['Subject'] = f"Inquiry Regarding {target_role} Opportunities at {firm_name} - Applicant"
    
    greeting = f"Dear {hr_name}," if hr_name != "Hiring Team" else f"To the Hiring Team at {firm_name},"
    body = ""
    match_score = "N/A"
    
    # Check if Gemini AI is active and not already known to be out of quota
    if resume_text and ai_client and not st.session_state.get('ai_quota_depleted', False):
        prompt = f"""
        You are an expert technical recruiter analyzing a candidate for a {target_role} role at {firm_name} near {location}.
        Based on the resume below:
        1. Give a 'Match Score' from 1-100 evaluating how well the candidate's skills fit this specific role.
        2. Write a professional cold email body (under 150 words) inquiring about the {target_role} position. Weave in 1 or 2 relevant skills from the resume. DO NOT include a subject line or greeting. Mention the attached resume. Sign off with "Best regards, [Your Name]".
        
        Reply strictly in this format:
        SCORE: [Insert Number]
        BODY:
        [Insert Email Body]
        
        Resume:
        {resume_text[:5000]}
        """
        
        for attempt in range(2):
            try:
                ai_response = ai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                response_text = ai_response.text.strip()
                if "SCORE:" in response_text and "BODY:" in response_text:
                    parts = response_text.split("BODY:")
                    match_score = parts[0].replace("SCORE:", "").strip()
                    body = f"{greeting}\n\n{parts[1].strip()}"
                else:
                    body = f"{greeting}\n\n{response_text}"
                
                if status_ui: status_ui.empty()
                break  
            except Exception as e:
                err_str = str(e).lower()
                # Detect Quota Exhaustion or Billing Limits immediately
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "billing" in err_str:
                    st.session_state.ai_quota_depleted = True
                    if status_ui:
                        status_ui.warning("🚨 Gemini AI Quota / Usage Credits Depleted! Falling back to standard high-converting template.")
                    body = None
                    break
                else:
                    if status_ui:
                        status_ui.warning(f"⚠️ Gemini Connection: Retrying... ({attempt+1}/2)")
                    time.sleep(2)
                    body = None

    if not body:
        body = f"""{greeting}\n\nI am writing to express my strong interest in joining {firm_name} as a {target_role}. \n\nGiven your excellent reputation in the {location} area, I would welcome the opportunity to contribute my skills to your team. I have attached my resume for your review.\n\nThank you for your time.\n\nBest regards,\n[Your Name]"""

    msg.set_content(body)
    safe_filename = "".join(x for x in firm_name if x.isalnum() or x in " _-") + ".eml"
    
    # Store draft in memory for instant multi-tenant ZIP generation
    if 'drafts_memory' not in st.session_state:
        st.session_state.drafts_memory = {}
    st.session_state.drafts_memory[safe_filename] = bytes(msg)
    
    # Save to user-isolated directory on disk (Multi-tenant safe)
    user_key = str(user_id) if user_id else (str(st.session_state.user.id) if ('user' in st.session_state and st.session_state.user) else "shared")
    user_drafts_dir = os.path.join(DRAFTS_DIR, user_key)
    os.makedirs(user_drafts_dir, exist_ok=True)
    file_path = os.path.join(user_drafts_dir, safe_filename)
    
    with open(file_path, 'wb') as f:
        f.write(bytes(msg))
        
    time.sleep(1.5)
    return match_score

def robust_get(url, headers, depth=0):
    if depth > 2: return None, None 
    try:
        try:
            response = requests.get(url, headers=headers, timeout=(4, 6), allow_redirects=True)
        except requests.exceptions.RequestException:
            if 'www.' in url:
                clean_url = url.replace('www.', '')
                response = requests.get(clean_url, headers=headers, timeout=(4, 6), allow_redirects=True)
            else:
                return None, None
                
        if response.status_code in [403, 406, 503]: return None, None
        
        html_text = response.text
        
        meta_refresh = re.search(r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\d+;\s*url=([^"\'>]+)["\']?', html_text, re.IGNORECASE)
        if meta_refresh:
            next_url = urljoin(response.url, meta_refresh.group(1))
            return robust_get(next_url, headers, depth + 1)
            
        js_redirect = re.search(r'window\.location\.replace\(["\']([^"\']+)["\']\)', html_text)
        if not js_redirect:
            js_redirect = re.search(r'window\.location\.href\s*=\s*["\']([^"\']+)["\']', html_text)
        if js_redirect:
            next_url = urljoin(response.url, js_redirect.group(1))
            return robust_get(next_url, headers, depth + 1)
        
        soup = BeautifulSoup(html_text, 'html.parser')
        
        if len(soup.get_text(strip=True)) < 50:
            iframe = soup.find('iframe')
            if iframe and iframe.get('src'):
                frame_url = urljoin(response.url, iframe['src'])
                return robust_get(frame_url, headers, depth + 1)
                
        return response, soup
    except Exception:
        return None, None

def get_hr_contact(website_url, search_query):
    if website_url == 'No website': return "Not Found", "Not Found"
    try:
        hint_words = [w.lower() for w in search_query.replace(',', '').split() if len(w) > 1]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        if not website_url.endswith('/'): website_url += '/'
        
        response, soup = robust_get(website_url, headers)
        if not response: return "Not Found", "Not Found"
        
        working_url = response.url
        urls_to_scrape = [working_url]
        target_keywords = ['team', 'about', 'contact', 'people', 'staff', 'directory', 'careers', 'professionals', 'leadership', 'our-team', 'recruiter', 'resources', 'center', 'locations']
        
        try:
            sitemap_resp = robust_get(urljoin(working_url, "sitemap.xml"), headers)
            if sitemap_resp[0] and sitemap_resp[0].status_code == 200:
                sitemap_urls = re.findall(r'<loc>(.*?)</loc>', sitemap_resp[0].text)
                for loc in sitemap_urls:
                    if any(kw in loc.lower() for kw in target_keywords):
                        if loc not in urls_to_scrape and not loc.endswith(('.pdf', '.jpg', '.png')):
                            urls_to_scrape.append(loc)
                            if len(urls_to_scrape) >= 8: break 
        except Exception:
            pass 
            
        team_pages = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').lower()
            link_text = a_tag.get_text(strip=True).lower()
            
            if any(keyword in href for keyword in target_keywords) or any(keyword in link_text for keyword in target_keywords):
                full_url = urljoin(working_url, a_tag['href'])
                if full_url not in urls_to_scrape and full_url.startswith('http') and not full_url.endswith(('.pdf', '.jpg')):
                    urls_to_scrape.append(full_url)
                    if any(kw in href for kw in ['team', 'people', 'leadership']):
                        team_pages.append(full_url)
                    if len(urls_to_scrape) >= 12: break 
        
        sub_profile_links = []
        for team_page in team_pages[:2]: 
            try:
                tp_resp, tp_soup = robust_get(team_page, headers)
                if not tp_soup: continue
                
                for a_tag in tp_soup.find_all('a', href=True):
                    href = a_tag.get('href', '').lower()
                    if ('team/' in href or 'people/' in href or 'profile' in href) and href not in [u.lower() for u in urls_to_scrape]:
                        full_url = urljoin(team_page, a_tag['href'])
                        sub_profile_links.append(full_url)
                        if len(sub_profile_links) >= 6: break 
            except Exception:
                continue
                
        urls_to_scrape.extend(sub_profile_links)
        
        combined_raw_text = ""
        for url in urls_to_scrape:
            try:
                page_resp, page_soup = robust_get(url, headers)
                if not page_resp: continue
                combined_raw_text += " " + page_resp.text + " "
                combined_raw_text += " " + page_soup.get_text(separator=' ', strip=True) + " "
            except Exception:
                continue
                
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        raw_emails = list(set(re.findall(email_pattern, combined_raw_text)))
        
        if not raw_emails: return "Not Found", "Not Found"
            
        text_lower = combined_raw_text.lower()
        email_scores = {}
        
        for raw_email in raw_emails:
            email = raw_email.lower().strip()
            
            if any(email.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.js', '.css', '.woff']):
                continue
            
            score = 0
            
            if any(kw in email for kw in ['contact', 'info', 'hello', 'admin', 'inquiries', 'support']):
                score += 200
                
            if any(kw in email for kw in ['hr@', 'careers@', 'talent@', 'recruiting@', 'recruiter@']):
                score += 100
                
            try:
                idx = text_lower.index(raw_email.lower())
                start_idx = max(0, idx - 150)
                end_idx = min(len(text_lower), idx + len(raw_email) + 150)
                window = text_lower[start_idx:end_idx]
                
                for word in hint_words:
                    if word in window:
                        score += 30
                        
                if any(title in window for title in ['partner', 'director', 'manager', 'acquisition', 'principal', 'president', 'consultant', 'specialist', 'administrator', 'advisor']):
                    score += 20
                    
            except ValueError:
                pass
                
            email_scores[email] = score
            
        if not email_scores: return "Not Found", "Not Found"
        
        best_email = max(email_scores, key=email_scores.get)
        
        if any(kw in best_email for kw in ['contact', 'info', 'hello', 'admin', 'inquiries', 'hr@', 'careers@', 'talent@', 'recruiting@', 'recruiter@']):
            contact_name = "Hiring Team"
        else:
            contact_name = "Employee Contact"
                
        return contact_name, best_email
        
    except Exception:
        return "Not Found", "Not Found"

# --- Custom Styling Overrides ---
st.markdown("""
    <style>
    .stApp {
        background-color: #F9FAFB;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 { color: #111827; font-weight: 600; letter-spacing: -0.015em; }
    
    [data-testid="stSidebar"] {
        background-color: #F3F4F6 !important;
        border-right: 1px solid #E5E7EB;
        padding-top: 10px !important;
    }
    
    .sidebar-heading {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: #4B5563 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        margin: 16px 0 8px 14px !important;
    }
    
    ::-webkit-scrollbar {
        width: 6px;
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #D1D5DB;
        border-radius: 3px;
    }
    
    div[data-testid="stSidebar"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
        width: calc(100% - 16px) !important;
        margin: 2px 8px !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        border: 1px solid #BFDBFE !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #DBEAFE !important;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #4B5563 !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #E5E7EB !important;
        color: #111827 !important;
    }
    
    .banner-container {
        width: 100%;
        height: 250px;
        overflow: hidden;
        border-radius: 12px;
        margin-top: 20px;
    }
    
    .metric-card {
        background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
        padding: 18px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 12px;
    }
    .metric-value { font-size: 30px; font-weight: 700; color: #111827; line-height: 1.2; }
    .metric-label { font-size: 12px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }
    
    .credit-badge {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white; padding: 14px 18px; border-radius: 8px; margin-bottom: 18px;
        display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .pricing-card {
        background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px;
        padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column; height: 100%;
    }
    .pricing-card.popular {
        border: 2px solid #2563EB; position: relative;
    }
    .pricing-badge {
        position: absolute; top: -12px; right: 18px; background-color: #2563EB; color: white;
        font-size: 11px; font-weight: 700; text-transform: uppercase; padding: 3px 10px; border-radius: 20px;
    }
    .pricing-title { font-size: 20px; font-weight: 700; color: #111827; margin-bottom: 6px; }
    .pricing-price { font-size: 36px; font-weight: 800; color: #111827; margin-bottom: 12px; }
    .pricing-feature { font-size: 14px; color: #4B5563; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
    
    .success-badge {
        background-color: #F8FAFC; color: #1E293B; padding: 12px 16px;
        border-radius: 6px; border: 1px solid #E2E8F0; font-size: 14px;
        font-weight: 500; display: flex; align-items: center; gap: 8px; margin-bottom: 15px;
        border-left: 4px solid #2563EB;
    }
    
    .stButton>button {
        background-color: #2563EB !important; color: white !important; border-radius: 6px !important;
        padding: 10px 20px !important; font-weight: 500 !important; border: none !important; transition: background-color 0.2s;
    }
    .stButton>button:hover { background-color: #1D4ED8 !important; }
    
    .clear-btn>button {
        background-color: #F3F4F6 !important; color: #4B5563 !important; border: 1px solid #D1D5DB !important;
        padding: 4px 12px !important; font-size: 12px !important;
    }
    .clear-btn>button:hover { background-color: #E5E7EB !important; }
    
    .guide-card {
        background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 24px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .guide-step { color: #2563EB; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    .guide-title { font-size: 19px; font-weight: 600; color: #111827; margin-bottom: 12px; }
    .guide-text { color: #4B5563; font-size: 15px; line-height: 1.6; }
    
    hr { border-color: #E5E7EB; margin-top: 5px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation Design ---
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
else:
    st.sidebar.title("LeadGenie")

st.sidebar.markdown(f'<div class="sidebar-heading">Account: {st.session_state.user.email}</div>', unsafe_allow_html=True)

# User Credit Balance Card
tier_label = st.session_state.tier.upper() if 'tier' in st.session_state else "FREE"
credits_val = st.session_state.credits_remaining if 'credits_remaining' in st.session_state else 10

st.sidebar.markdown(f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px; margin: 0 8px 10px 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:11px; font-weight:700; color:#6B7280; text-transform:uppercase; letter-spacing:0.05em;">Balance</span>
            <span style="background-color:#DBEAFE; color:#1D4ED8; font-size:10px; font-weight:700; padding:2px 8px; border-radius:12px;">{tier_label}</span>
        </div>
        <div style="font-size: 22px; font-weight: 700; color: #111827; margin: 4px 0;">{credits_val} <span style="font-size:12px; font-weight:500; color:#6B7280;">Leads Left</span></div>
    </div>
""", unsafe_allow_html=True)

# Real-Time Gemini AI Quota Indicator in Sidebar
ai_status_code, ai_status_desc, ai_status_icon = check_gemini_quota_status()
if st.session_state.get('ai_quota_depleted', False) or ai_status_code in ["QUOTA_EXHAUSTED", "BILLING_REQUIRED"]:
    ai_sidebar_text = "🔴 AI: Quota Exhausted"
    ai_sidebar_color = "#DC2626"
elif ai_status_code == "OPERATIONAL":
    ai_sidebar_text = "🟢 AI: Operational"
    ai_sidebar_color = "#059669"
elif ai_status_code == "HIGH_DEMAND":
    ai_sidebar_text = "🟡 AI: High Demand Spike"
    ai_sidebar_color = "#D97706"
else:
    ai_sidebar_text = f"{ai_status_icon} AI: {ai_status_code}"
    ai_sidebar_color = "#DC2626"

st.sidebar.markdown(f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 10px 12px; margin: 0 8px 14px 8px;">
        <div style="font-size:10px; font-weight:700; color:#6B7280; text-transform:uppercase; letter-spacing:0.05em;">AI Quota Health</div>
        <div style="font-size:12px; font-weight:700; color:{ai_sidebar_color}; margin-top:2px;">{ai_sidebar_text}</div>
    </div>
""", unsafe_allow_html=True)

if st.sidebar.button("Dashboard", key="nav_dash", type="primary" if st.session_state.page == "Dashboard" else "secondary", use_container_width=True):
    st.session_state.page = "Dashboard"
    st.rerun()

if st.sidebar.button("Lead Database", key="nav_db", type="primary" if st.session_state.page == "Lead Database" else "secondary", use_container_width=True):
    st.session_state.page = "Lead Database"
    st.rerun()

if st.sidebar.button("Pricing & Plans", key="nav_pricing", type="primary" if st.session_state.page == "Pricing & Plans" else "secondary", use_container_width=True):
    st.session_state.page = "Pricing & Plans"
    st.rerun()

if st.sidebar.button("How to Use & Examples", key="nav_guide", type="primary" if st.session_state.page == "How to Use & Examples" else "secondary", use_container_width=True):
    st.session_state.page = "How to Use & Examples"
    st.rerun()

if st.sidebar.button("About", key="nav_about", type="primary" if st.session_state.page == "About" else "secondary", use_container_width=True):
    st.session_state.page = "About"
    st.rerun()

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Logout", key="nav_logout", type="secondary", use_container_width=True):
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("v1.2.0 | Members-Only Commercial Edition")


# --- Page 1: Dashboard ---
if st.session_state.page == "Dashboard":
    
    if st.session_state.show_success_banner:
        st.success(f"Pipeline Execution Complete. {st.session_state.last_processed_count} leads successfully generated. Please proceed to the Lead Database to review and export.")
        st.session_state.show_success_banner = False 
    
    st.title("Outreach Dashboard")
    st.markdown("Automate your lead generation pipeline by extracting decision-maker contacts and synthesizing custom outreach files.")
    
    # Active Gemini Quota Exhaustion Banner
    if st.session_state.get('ai_quota_depleted', False) or ai_status_code in ["QUOTA_EXHAUSTED", "BILLING_REQUIRED"]:
        st.error("""
            🚨 **Notice to Users & Admin:** The connected Google Gemini AI engine has reached its quota limit or is out of usage credits.
            * The pipeline will automatically generate structured high-converting template drafts without failing, but AI resume-match personalization is temporarily paused until credits are refilled in [Google AI Studio](https://aistudio.google.com/).
        """)
        
    st.caption("ℹ️ **Disclaimer:** LeadGenie actively scans domain architecture for public contact information. Organizations that strictly utilize gated web forms or conceal contact data will be flagged in the database as requiring manual submission.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.3, 1])
    
    with col_left:
        st.subheader("Search Configuration")
        with st.container(border=True):
            
            # Live Credit Status Indicator
            if st.session_state.credits_remaining <= 0:
                st.error("🚨 **Out of Credits:** You have 0 leads remaining. Please upgrade your plan to run searches.")
            elif st.session_state.credits_remaining < st.session_state.input_num:
                st.warning(f"⚠️ **Low Balance:** You have **{st.session_state.credits_remaining}** leads remaining, but requested {st.session_state.input_num}. The pipeline will stop once credits run out.")
            else:
                st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding:10px 14px; background-color:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px;">
                        <span style="font-weight:600; color:#1E40AF; font-size:13px;">⚡ Search Credits: <b>{st.session_state.credits_remaining} Remaining</b></span>
                        <span style="font-size:12px; color:#2563EB; font-weight:600;">Plan: {st.session_state.tier.upper()}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Quick-Activation Box for existing users on the Free tier
            if st.session_state.tier == 'free':
                with st.expander("🔑 Have a License Key or want to claim 300 VIP Leads?", expanded=True):
                    st.markdown("Enter `GENIE-VIP` below to instantly upgrade this existing account to 300 leads:")
                    c_code, c_btn = st.columns([3, 1])
                    with c_code:
                        dash_code = st.text_input("License Key", value="GENIE-VIP", key="dash_promo_code", label_visibility="collapsed").strip().upper()
                    with c_btn:
                        if st.button("Apply", key="dash_apply_btn", use_container_width=True):
                            is_val, msg, add_credits, add_tier = verify_lemon_squeezy_license(dash_code, str(st.session_state.user.email))
                            if is_val:
                                st.session_state.credits_remaining = add_credits
                                st.session_state.tier = add_tier
                                try:
                                    supabase.table('user_profiles').update({
                                        'credits_remaining': st.session_state.credits_remaining,
                                        'tier': add_tier
                                    }).eq('user_id', st.session_state.user.id).execute()
                                except Exception:
                                    pass
                                st.success(f"🎉 Upgraded! You now have {add_credits} Leads ({add_tier.upper()}).")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")

            search_query = st.text_input("Target Location & Industry", value=st.session_state.input_location)
            
            col_inner1, col_inner2 = st.columns(2)
            with col_inner1:
                target_role = st.text_input("Target Role", value=st.session_state.input_role)
            with col_inner2:
                default_idx = 0 if st.session_state.input_num == 5 else 1
                num_leads = st.selectbox("Leads to Generate", options=[5, 10], index=default_idx)
                
            st.session_state.input_location = search_query
            st.session_state.input_role = target_role
            st.session_state.input_num = num_leads
                
            st.markdown("---")
            st.markdown("**Resume Context (Recommended)**")
            
            if st.session_state.resume_text != "":
                st.markdown(f'<div class="success-badge">📄 <b>Attached File:</b> {st.session_state.resume_filename}</div>', unsafe_allow_html=True)
                
                _, clear_col = st.columns([4, 1])
                with clear_col:
                    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
                    if st.button("Clear Resume", use_container_width=True):
                        st.session_state.resume_text = ""
                        st.session_state.resume_filename = ""
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                uploaded_file = st.file_uploader("Upload PDF or TXT to enable profile matching and dynamic email synthesis.", type=["pdf", "txt"], label_visibility="collapsed")
                
                if uploaded_file is not None:
                    st.session_state.resume_text = extract_resume_text(uploaded_file)
                    st.session_state.resume_filename = uploaded_file.name
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action Button: Guarded by credit balance
            is_out_of_credits = (st.session_state.credits_remaining <= 0)
            if is_out_of_credits:
                submit_button = False
                if st.button("💳 Upgrade Plan to Continue", type="primary", use_container_width=True):
                    st.session_state.page = "Pricing & Plans"
                    st.rerun()
            else:
                submit_button = st.button("Execute Pipeline", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Session Telemetry")
        
        try:
            telemetry_req = supabase.table('leads').select('*', count='exact').eq('user_id', st.session_state.user.id).execute()
            total_db_leads = len(telemetry_req.data) if telemetry_req.data else 0
            
            success_count = sum(1 for lead in telemetry_req.data if lead['email'] not in ["Not Found", "Error", "API Overload", "N/A"]) if total_db_leads > 0 else 0
            success_rate = round((success_count / total_db_leads * 100) if total_db_leads > 0 else 0)
        except Exception:
            total_db_leads, success_count, success_rate = 0, 0, 0
        
        st.markdown(f"""
            <div style="display: flex; flex-direction: column;">
                <div class="metric-card">
                    <div class="metric-value">{total_db_leads}</div>
                    <div class="metric-label">Aggregated Leads</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{success_count}</div>
                    <div class="metric-label">Verified Contacts Extracted</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{success_rate}%</div>
                    <div class="metric-label">Extraction Accuracy Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 18px; color:{ai_sidebar_color};">{ai_sidebar_text}</div>
                    <div class="metric-label">Gemini AI Quota Health</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # AI Health diagnostic trigger
        with st.expander("🩺 Test Gemini AI Connection"):
            if st.button("Run Diagnostic Ping", use_container_width=True):
                with st.spinner("Pinging Gemini API..."):
                    try:
                        t0 = time.time()
                        ping_res = ai_client.models.generate_content(model=GEMINI_MODEL, contents="Hello")
                        elapsed = round((time.time() - t0) * 1000)
                        st.session_state.ai_quota_depleted = False
                        st.success(f"✅ AI Engine Healthy! Latency: {elapsed}ms. Quota and credits are active.")
                    except Exception as e:
                        st.error(f"❌ Connection Failure: {e}")

        status_ui = st.empty()
        progress_ui = st.empty()

    if submit_button:
        st.warning("⚠️ **Pipeline Active:** Please remain on this tab until completion. Navigating to another tab will cancel the current search.")
        
        try:
            supabase.table('user_profiles').update({
                'target_location': search_query,
                'target_role': target_role,
                'leads_to_generate': num_leads
            }).eq('user_id', st.session_state.user.id).execute()
        except Exception:
            pass
            
        # Multi-tenant safe file initialization: isolate by user ID
        user_key = str(st.session_state.user.id)
        user_drafts_dir = os.path.join(DRAFTS_DIR, user_key)
        if os.path.exists(user_drafts_dir):
            shutil.rmtree(user_drafts_dir)
        os.makedirs(user_drafts_dir, exist_ok=True)
        
        if 'drafts_memory' not in st.session_state:
            st.session_state.drafts_memory = {}
        st.session_state.drafts_memory.clear()
        
        st.session_state.total_searches += 1
        
        status_ui.info(f"Querying Google Places API for: {search_query}...")
        progress_bar = progress_ui.progress(0.0)
        
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": PLACES_API_KEY, "X-Goog-FieldMask": "places.displayName.text,places.formattedAddress,places.websiteUri"}
        response = requests.post(url, headers=headers, json={"textQuery": search_query, "maxResultCount": 20})
        
        if response.status_code != 200:
            st.error("API Error. Verify system configuration.")
        else:
            places = response.json().get('places', [])
            processed = 0
            
            try:
                existing_db = supabase.table('leads').select('website').eq('user_id', st.session_state.user.id).execute()
                saved_websites = [row['website'] for row in existing_db.data] if existing_db.data else []
            except Exception:
                saved_websites = []
            
            for idx, place in enumerate(places):
                if processed >= num_leads: break
                if st.session_state.credits_remaining <= 0:
                    status_ui.warning("⚡ Credit limit reached during search. Ending batch.")
                    break
                
                name = place.get('displayName', {}).get('text', 'Unknown Name')
                address = place.get('formattedAddress', 'No address')
                website = place.get('websiteUri', 'No website')
                city_location = address.split(",")[1].strip() if "," in address else "your area"
                
                if website != 'No website':
                    if website in saved_websites: continue
                        
                    status_ui.info(f"Deep scraping website architecture: {name}...")
                    progress_bar.progress((processed + 0.2) / num_leads)
                    
                    hr_name, hr_email = get_hr_contact(website, search_query)
                    
                    if hr_email in ["Not Found", "Error", "API Overload", "N/A"]:
                        notes, hr_name, hr_email = "Requires manual website submission.", "Not Found", "Not Found"
                        match_score = "N/A"
                    else:
                        notes = ""
                        status_ui.info(f"AI synthesizing personalized outreach for: {name}...")
                        progress_bar.progress((processed + 0.6) / num_leads)
                        
                        match_score = create_eml_draft(name, hr_name, hr_email, city_location, target_role, st.session_state.resume_text, user_id=st.session_state.user.id, status_ui=status_ui)
                        st.session_state.successful_emails += 1
                    
                    try:
                        supabase.table('leads').insert({
                            "user_id": st.session_state.user.id,
                            "firm_name": name,
                            "location": address,
                            "website": website,
                            "contact_name": hr_name,
                            "email": hr_email,
                            "status": "Not Contacted",
                            "notes": notes,
                            "match_score": match_score
                        }).execute()
                    except Exception:
                        pass
                    
                    # Deduct 1 credit per processed lead & sync with Supabase
                    processed += 1
                    st.session_state.credits_remaining = max(0, st.session_state.credits_remaining - 1)
                    try:
                        supabase.table('user_profiles').update({
                            'credits_remaining': st.session_state.credits_remaining
                        }).eq('user_id', st.session_state.user.id).execute()
                    except Exception:
                        pass
                    
                    progress_bar.progress(processed / num_leads)
                    st.session_state.last_processed_count = processed
                    
            status_ui.empty()
            progress_ui.empty()
            st.session_state.show_success_banner = True
            st.rerun()

# --- Page 2: Lead Database (Interactive Layout) ---
elif st.session_state.page == "Lead Database":
    st.title("Interactive Lead Database")
            
    st.markdown("Manage your outreach pipeline directly within the platform. Leads are automatically synced from your cloud account.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        db_leads = supabase.table('leads').select('*').eq('user_id', st.session_state.user.id).execute()
        db_data = db_leads.data
    except Exception:
        db_data = []
    
    if db_data and len(db_data) > 0:
        master_df = pd.DataFrame(db_data)
        display_cols = ["firm_name", "location", "website", "contact_name", "email", "status", "notes", "match_score"]
        ui_df = master_df[display_cols]
        
        ui_df.columns = ["Firm Name", "Location", "Website", "Contact Name", "Email", "Status", "Notes", "Match Score"]
        
        edited_df = st.data_editor(
            ui_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Pipeline Status",
                    help="Track your outreach progress.",
                    options=["Not Contacted", "Emailed", "Followed Up", "Interviewing", "Rejected", "Offer Extended"],
                    required=True,
                ),
                "Website": st.column_config.LinkColumn("Website"),
                "Match Score": st.column_config.TextColumn("Match Score", disabled=True),
            }
        )
        
        st.markdown("### File Export")
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='LeadGenie_CRM')
                worksheet = writer.sheets['LeadGenie_CRM']
                
                for idx, col in enumerate(worksheet.columns, 1):
                    worksheet.column_dimensions[col[0].column_letter].width = 25
                
                worksheet.freeze_panes = 'A2'
                
            st.download_button(
                label="Download CRM (.xlsx)", 
                data=buffer.getvalue(), 
                file_name="LeadGenie_Cloud_CRM.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                use_container_width=True
            )
            
        with col2:
            # Multi-tenant safe: export from session memory or user's isolated folder
            user_key = str(st.session_state.user.id)
            user_drafts_dir = os.path.join(DRAFTS_DIR, user_key)
            has_memory_drafts = ('drafts_memory' in st.session_state and len(st.session_state.drafts_memory) > 0)
            has_disk_drafts = (os.path.exists(user_drafts_dir) and len(os.listdir(user_drafts_dir)) > 0)
            
            if has_memory_drafts or has_disk_drafts:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    if has_memory_drafts:
                        for fname, fbytes in st.session_state.drafts_memory.items():
                            zip_file.writestr(fname, fbytes)
                    elif has_disk_drafts:
                        for file in os.listdir(user_drafts_dir):
                            file_path = os.path.join(user_drafts_dir, file)
                            if os.path.isfile(file_path):
                                zip_file.write(file_path, file)
                                
                st.download_button(
                    label="Download Emails (.zip)", 
                    data=zip_buffer.getvalue(), 
                    file_name="LeadGenie_Drafts.zip", 
                    mime="application/zip", 
                    use_container_width=True
                )
            else:
                st.button("No Drafts Available", disabled=True, use_container_width=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Clear Session Cache", type="secondary"):
            st.session_state.total_searches = 0
            st.session_state.successful_emails = 0
            st.session_state.resume_text = ""
            st.session_state.resume_filename = ""
            if 'drafts_memory' in st.session_state:
                st.session_state.drafts_memory.clear()
            user_key = str(st.session_state.user.id)
            user_drafts_dir = os.path.join(DRAFTS_DIR, user_key)
            if os.path.exists(user_drafts_dir): 
                shutil.rmtree(user_drafts_dir)
            st.rerun()
    else:
        st.info("Your cloud database is currently empty. Execute a search on the Dashboard to populate data.")

# --- Page 3: Pricing & Plans (Lemon Squeezy Integration) ---
elif st.session_state.page == "Pricing & Plans":
    st.title("Pricing & Credit Packs")
    st.markdown("Unlock more verified decision-maker contacts and AI-crafted cold emails to supercharge your internship recruitment.")
    st.markdown("---")
    
    # Balance Banner
    current_tier = st.session_state.tier.upper()
    current_credits = st.session_state.credits_remaining
    st.markdown(f"""
        <div class="credit-badge">
            <div>
                <div style="font-size: 13px; opacity: 0.9; text-transform: uppercase; font-weight: 600;">Active Account Balance</div>
                <div style="font-size: 26px; font-weight: 800;">{current_credits} Leads Available</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; opacity: 0.9;">Plan Level</div>
                <div style="font-size: 18px; font-weight: 700; background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 6px;">{current_tier}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Construct Lemon Squeezy Checkout URLs with user metadata
    lemon_user_id = str(st.session_state.user.id)
    lemon_email = str(st.session_state.user.email)
    
    starter_checkout_url = f"{LEMON_SQUEEZY_STARTER_URL}?checkout[custom][user_id]={lemon_user_id}&checkout[email]={lemon_email}" if LEMON_SQUEEZY_STARTER_URL else "https://lemonsqueezy.com"
    pro_checkout_url = f"{LEMON_SQUEEZY_PRO_URL}?checkout[custom][user_id]={lemon_user_id}&checkout[email]={lemon_email}" if LEMON_SQUEEZY_PRO_URL else "https://lemonsqueezy.com"
    
    plan_col1, plan_col2, plan_col3 = st.columns(3)
    
    with plan_col1:
        st.markdown("""
            <div class="pricing-card">
                <div class="pricing-title">Trial Level</div>
                <div style="color: #6B7280; font-size: 13px; margin-bottom: 8px;">Explore LeadGenie</div>
                <div class="pricing-price">$0</div>
                <hr>
                <div class="pricing-feature">✅ 10 Verified Leads</div>
                <div class="pricing-feature">✅ Decision-Maker Regex Scraper</div>
                <div class="pricing-feature">✅ AI Match Score (1-100)</div>
                <div class="pricing-feature">✅ Ready-to-Send .eml Payloads</div>
                <div class="pricing-feature">✅ Interactive CRM Database</div>
                <div style="margin-top: auto; padding-top: 20px;">
                    <div style="background-color: #F3F4F6; text-align: center; padding: 10px; border-radius: 6px; font-size: 14px; font-weight: 600; color: #4B5563;">
                        Current / Base Tier
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with plan_col2:
        st.markdown("""
            <div class="pricing-card popular">
                <div class="pricing-badge">Most Popular</div>
                <div class="pricing-title">Job Hunter Pack</div>
                <div style="color: #6B7280; font-size: 13px; margin-bottom: 8px;">For active internship searchers</div>
                <div class="pricing-price">$19 <span style="font-size: 14px; font-weight: 500; color: #6B7280;">one-time</span></div>
                <hr>
                <div class="pricing-feature">✅ <b>100 Verified Leads</b></div>
                <div class="pricing-feature">✅ Priority Headless Domain Scraping</div>
                <div class="pricing-feature">✅ Contextual Resume-Aligned Emails</div>
                <div class="pricing-feature">✅ Full Master CRM (.xlsx) Export</div>
                <div class="pricing-feature">✅ Direct Desktop Mail (.eml) ZIP</div>
                <div class="pricing-feature">✅ Instant License Key Delivery</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("⚡ Buy 100 Leads ($19)", starter_checkout_url, type="primary", use_container_width=True)

    with plan_col3:
        st.markdown("""
            <div class="pricing-card">
                <div class="pricing-title">Pro Career Hunter</div>
                <div style="color: #6B7280; font-size: 13px; margin-bottom: 8px;">Maximum outreach volume</div>
                <div class="pricing-price">$39 <span style="font-size: 14px; font-weight: 500; color: #6B7280;">one-time</span></div>
                <hr>
                <div class="pricing-feature">✅ <b>300 Verified Leads</b></div>
                <div class="pricing-feature">✅ Highest Priority Deep Web Crawling</div>
                <div class="pricing-feature">✅ Multi-Metro Outreach Campaigns</div>
                <div class="pricing-feature">✅ Gemini 3.5 AI Synthesis</div>
                <div class="pricing-feature">✅ Lifetime Cloud Access</div>
                <div class="pricing-feature">✅ VIP Fast-Track Support</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("🚀 Buy 300 Leads ($39)", pro_checkout_url, type="secondary", use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Instant License / Promo Code Redemption
    with st.expander("🔑 Have a Promo Code or Lemon Squeezy Order #? Click to Redeem", expanded=False):
        st.markdown("If you completed checkout or received an additional promotional grant, enter your activation key below to immediately add leads to your account.")
        code_col, btn_col = st.columns([3, 1])
        with code_col:
            promo_code = st.text_input("Activation / Promo Key", placeholder="e.g. GENIE-VIP or Order Ref", label_visibility="collapsed").strip().upper()
        with btn_col:
            if st.button("Redeem Code", use_container_width=True):
                is_val, msg, add_credits, add_tier = verify_lemon_squeezy_license(promo_code, str(st.session_state.user.email))
                if is_val:
                    st.session_state.credits_remaining += add_credits
                    st.session_state.tier = add_tier
                    try:
                        supabase.table('user_profiles').update({
                            'credits_remaining': st.session_state.credits_remaining,
                            'tier': add_tier
                        }).eq('user_id', st.session_state.user.id).execute()
                    except Exception:
                        pass
                    st.success(f"🎉 {msg} +{add_credits} Leads added to your account!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

    # Lemon Squeezy Configuration Guide for Store Owner
    if not LEMON_SQUEEZY_STARTER_URL:
        with st.expander("⚙️ Admin Setup: How to connect your Lemon Squeezy store"):
            st.markdown("""
            1. Create a free product at [LemonSqueezy.com](https://lemonsqueezy.com).
            2. Enable **Generate license keys** in product settings.
            3. Copy your product checkout share URLs.
            4. Paste them into `.streamlit/secrets.toml` or Streamlit Cloud Secrets:
            ```toml
            LEMON_SQUEEZY_STARTER_URL = "https://yourstore.lemonsqueezy.com/buy/variant-id-100"
            LEMON_SQUEEZY_PRO_URL = "https://yourstore.lemonsqueezy.com/buy/variant-id-300"
            ```
            """)

# --- Page 4: How to Use & Examples ---
elif st.session_state.page == "How to Use & Examples":
    st.title("User Guide & Examples")
    st.markdown("Follow these steps to maximize your outbound conversion rate using LeadGenie.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="guide-card">
            <div class="guide-step">Step 1</div>
            <div class="guide-title">Configure Your Search</div>
            <div class="guide-text">
                Input highly specific, localized parameters on the Dashboard. Instead of broad searches like 'Accounting Firms in New York', use narrower scopes such as <b>'Boutique Accounting Firms in White Plains, NY'</b>. The engine performs best when returning localized, mid-sized businesses.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="guide-card">
            <div class="guide-step">Step 2</div>
            <div class="guide-title">Upload Context</div>
            <div class="guide-text">
                Upload your resume in PDF or TXT format. LeadGenie's LLM engine cross-references your background against the scraped profile of each firm, allowing it to weave highly specific details into the email body, proving to the recipient that the outreach is customized.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
        
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="guide-card">
            <div class="guide-step">Step 3</div>
            <div class="guide-title">Execute & Review</div>
            <div class="guide-text">
                Click 'Execute Pipeline'. The application will dispatch a headless scraper to traverse the domain architecture of the generated leads. The algorithm uses Contextual Proximity Scoring to prioritize decision-maker and HR emails over generic employee inboxes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="guide-card">
            <div class="guide-step">Step 4</div>
            <div class="guide-title">Track & Export</div>
            <div class="guide-text">
                Navigate to the Lead Database. You can edit notes and update the pipeline status of each lead directly in the web app. Download the finalized CRM CSV file to save your progress locally, and extract the generated `.eml` payloads via the ZIP download.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Expected Output Reference Examples")
    st.markdown("Preview the format and accuracy of dataset outputs from the active session pipeline:")
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Example 1: Master CRM Database Format", expanded=True):
        st.markdown("LeadGenie generates a structured dataset ready for immediate CRM import. It actively tracks pipeline status, contextual notes, and algorithmic match scores.")
        example_data = [
            ["Apex Financial Advisors", "New York, NY", "apexfinancial.com", "Hiring Team", "info@apexfinancial.com", "Not Contacted", "", "88"],
            ["Blue Ridge Consulting", "Stamford, CT", "blueridge.com", "Employee Contact", "michael@blueridge.com", "Emailed", "Location proximity bonus applied.", "92"],
            ["Summit Accounting", "White Plains, NY", "summitcpa.com", "Not Found", "Not Found", "Not Contacted", "Requires manual website submission.", "N/A"]
        ]
        st.dataframe(pd.DataFrame(example_data, columns=["Firm Name", "Location", "Website", "Contact Name", "Email", "Status", "Notes", "Match Score"]), use_container_width=True)
    
    with st.expander("Example 2: Synthesized EML Payload"):
        st.markdown("The `.eml` files generated by LeadGenie are designed to be natively opened in standard desktop mail clients (like Outlook or Apple Mail), pre-addressed and formatted for immediate review.")
        st.code("""
To: info@apexfinancial.com
Subject: Inquiry Regarding Audit Intern Opportunities at Apex Financial Advisors - Applicant

To the Hiring Team at Apex Financial Advisors,

I am writing to express my strong interest in joining Apex Financial Advisors as an Audit Intern. 
With my coursework in Financial and Managerial Accounting and my hands-on experience managing ledgers, 
I am highly prepared to contribute to your practice.

Given your excellent reputation in the New York area, I would welcome the opportunity to discuss how my 
analytical background aligns with your team's objectives. I have attached my resume for your review.

Thank you for your time and consideration.

Best regards,
[Your Name]
""", language="email")
        
    with st.expander("Example 3: Contextual Proximity Engine (Under the Hood)"):
        st.markdown("""
        LeadGenie utilizes a tiered Regex scoring algorithm to ensure the highest-value contact is extracted from a domain:
        
        *   **Priority 1 (Base Score: 200):** General contact and administration inboxes (`contact@`, `info@`, `hello@`).
        *   **Priority 2 (Base Score: 100):** Dedicated talent acquisition inboxes (`hr@`, `careers@`, `recruiting@`).
        *   **Priority 3 (Employee Context Bonus):** If only individual employee emails are found, the engine scans a 300-character window surrounding the email on the live webpage.
            *   *Location Match:* +30 Points (e.g., The email sits adjacent to the targeted search location).
            *   *Title Match:* +20 Points (e.g., The email sits adjacent to keywords like 'Partner', 'Director', or 'Principal').
        """)

# --- Page 5: About ---
elif st.session_state.page == "About":
    st.title("About LeadGenie")
    st.markdown("Learn more about the custom, autonomous job-hunting outreach partner.")
    st.markdown("---")
    
    st.subheader("Why LeadGenie Was Created")
    
    st.markdown("""
    LeadGenie was created as a personal endeavor to streamline and automate the incredibly manual, time-consuming process of internship and job hunting. 
    
    Navigating the highly competitive landscape of corporate roles often requires spending countless hours browse-hunting local, boutique, or mid-sized firms on maps, analyzing website architecture, scouring 'About' or 'Team' pages for contact directories, and manually draft-emailing recruiters and partners. It became clear this workflow could be heavily optimized by autonomous code. 
    
    By combining map-based geo-targeting, dynamic web scraping, context-based email filtering (matching targeted hiring teams over generic lists), and personalized AI drafting using natural matching logic, LeadGenie does the heavy lifting. It converts search volume into actionable, personalized, and easily trackable database outreach. In short, it was built to transform networking from a tedious chore into an intelligent, data-driven system—saving hundreds of hours in the search for professional opportunities.
    """)
    
    if os.path.exists("banner.png"):
        import base64
        with open("banner.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(f"""
            <div class="banner-container">
                <img src="data:image/png;base64,{encoded_string}" style="width:100%; height:100%; object-fit:cover; border-radius:12px;">
            </div>
        """, unsafe_allow_html=True)
