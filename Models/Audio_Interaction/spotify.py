import spotipy
from spotipy.oauth2 import SpotifyOAuth
import sys
import webbrowser

# Configuration
CLIENT_ID = "6154b55186544a898babdd648f511332"
CLIENT_SECRET = "8655cff17fab439d84cf3f89b2dab849"  # TODO: Paste your secret here
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPE = "user-library-read"  # Changed scope since we can't control playback without Premium

def get_spotify_client():
    """Authenticate and return a Spotify client."""
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    ))

def play_music(query):
    """Search for a track and open it in the browser (works for Free accounts)."""
    sp = get_spotify_client()
    
    # Search for the track - improved query
    # We search for type 'track' and take the first result.
    # Adding 'track:' prefix can sometimes help if the query is ambiguous, 
    # but usually a direct search is best.
    print(f"Searching for: '{query}'")
    results = sp.search(q=query, limit=5, type='track')
    tracks = results['tracks']['items']
    
    if not tracks:
        return f"No tracks found for '{query}'."
    
    # Simple heuristic: Pick the first one, but print top 3 for debugging
    print("Top results:")
    for i, t in enumerate(tracks[:3]):
        print(f"{i+1}: {t['name']} by {t['artists'][0]['name']}")

    track = tracks[0]
    track_name = track['name']
    artist_name = track['artists'][0]['name']
    track_url = track['external_urls']['spotify']
    
    print(f"Selected: {track_name} by {artist_name}")
    print(f"Opening: {track_url}")
    
    # Open the track in the default web browser
    webbrowser.open(track_url)
    
    return f"Opening '{track_name}' by {artist_name} in Spotify..."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(play_music(query))
    else:
        print("Usage: python spotify.py [song name]")