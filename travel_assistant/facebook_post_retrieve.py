from langgraph.store.base import BaseStore


def get_friends_posts(store: BaseStore, user_id: str):
    print("\n===== DEBUG: FETCH FRIEND POSTS =====")

    mapping = store.get(("user_mapping", user_id), "fb_id")
    print("User mapping:", mapping.value if mapping else None)

    if not mapping:
        print("❌ No FB mapping found")
        return []

    fb_user_id = mapping.value

    friends_obj = store.get(("fb_friends", fb_user_id), "list")
    print("Friends list:", friends_obj.value if friends_obj else None)

    if not friends_obj:
        print("❌ No friends found")
        return []

    friend_ids = friends_obj.value
    all_posts = []

    for fid in friend_ids:
        posts_obj = store.get(("fb_posts", fid), "posts")
        print(f"Posts for {fid}:", posts_obj.value if posts_obj else None)

        if not posts_obj:
            continue

        for p in posts_obj.value:
            all_posts.append({
                "friend_fb_id": fid,
                "message": p.get("message"),
                "place_name": p.get("place_name"),
                "place_city": p.get("place_city"),
                "place_country": p.get("place_country"),
                "place_lat": p.get("place_lat"),
                "place_lng": p.get("place_lng"),
                "created_time": p.get("created_time")
            })

    print(f"✅ Total posts collected: {len(all_posts)}")
    print("====================================\n")

    return all_posts