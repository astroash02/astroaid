"""
Navigation bar component for AstroAid Dashboard
"""

import dash_bootstrap_components as dbc
from dash import html
import config

def create_navbar():
    """Create the main navigation bar"""
    
    navbar = dbc.Navbar([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src="/assets/logo.png", 
                        height="30px",
                        className="me-2"
                    ),
                    dbc.NavbarBrand(
                        config.BRAND_NAME, 
                        className="ms-2 fw-bold"
                    ),
                ], width="auto"),
                
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/", active=True)),
                        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics")),
                        dbc.NavItem(dbc.NavLink("ML Models", href="/ml")),
                        dbc.NavItem(dbc.NavLink("Help", href="/help")),
                    ], navbar=True, className="ms-auto")
                ])
            ], align="center", className="w-100")
        ], fluid=True)
    ], 
    color="dark", 
    dark=True, 
    className="mb-4"
    )
    
    return navbar
