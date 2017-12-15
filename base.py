from rackspace import connection

conn = connection.Connection(username="thelightbox", api_key="54c8d6ce9c284a23ac174a4dfdb05ba3", region="DFW")
for cont in conn.object_store.containers():
    print(cont.name)
    if cont.name == "hotpoint-cdn":
        for obj in conn.object_store.objects(cont):
            objname = str(obj.name)
            if objname.startswith("images/videothumbs"):
                print(objname)

