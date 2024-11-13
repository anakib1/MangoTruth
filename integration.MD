## Classroom 

```

get-files (userId:uuid, fromTime:timestamp, addition_args : Dict)

addition_args = { "classroomId" : " 00001111=" }

Response:
    List[Assignment]
    List[Submission]
    Status : ["SUCCESS", "NO_CREDENTIAL", "INTERNAL_ERROR"]

List[Assignment]
- sorted by timetamp 
- timestamp
- at most 100 assignments

Assignment (from teacher):  
- assignmentId
- creationTime
- modificationTime
- deadlineTime
- Content
- url

Submission (from student) 
- submissionId
- assignmentId
- submissionTime
- userId 
- content : byte
- contentType : ["text", "pdf", "docx", "txt"]

```
```
src/classroom_engine.py 
src/communication/communicator.py:
    - add_listener
    - write_response
src/communication/rabbit.py 
src/communication/rest.py 

classroom_engine:
    commun = createCommunicator()
    commun.add_listener(on_request)
    
    on_request():
        raw_resp = requets.submit("classroom/api/assignments, ???")
        resp : List[Assignment] = convert(resp) 
        commun.write_response(resp)
      
rabbit:
    while true:
        obj = read_from_queue()
        listener.call(obj)
    
    

```



```
Black baclek jfdjgs <FF00FF> kjgfskjgjdgsojgsjn <000000>
```