"""Launch the browser-based Motor Intelligence Center.

The dashboard UI is implemented by the Flask API and HTML frontend in
``web_dashboard``.  This compatibility entry point keeps the existing
``python app/06_dashboard.py`` command while removing the Streamlit
runtime dependency.
"""

from web_dashboard.server import app


def main():
    """Start the dashboard web server."""
    print()
    print("=" * 56)
    print("  MOTOR INTELLIGENCE CENTER")
    print("  Dashboard: http://localhost:5000")
    print("=" * 56)
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
