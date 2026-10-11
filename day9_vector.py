import chromadb

# 1. Create the vector database (saved in a local folder called chroma_db)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="university_faqs")

# 2. Our information: (id, category, title, text)
faqs = [
    ("faq1", "Admissions", "Admission Dates",
     "Admission applications for the fall semester open on March 1 and close on June 30."),
    ("faq2", "Admissions", "Eligibility",
     "Applicants need at least 60 percent marks in their previous degree and must pass the university entry test."),
    ("faq3", "Fees", "Fee Payment",
     "The tuition fee is paid per semester, and students can choose to pay in two installments."),
    ("faq4", "Scholarships", "Merit Scholarship",
     "Students with a GPA above 3.5 can apply for a merit scholarship that reduces tuition by up to 50 percent."),
    ("faq5", "Hostel", "Hostel Facilities",
     "On-campus hostels are available for both male and female students, and rooms can be booked after admission is confirmed."),
    ("faq6", "Library", "Library Hours",
     "The library is open from 8 AM to 10 PM on weekdays and offers free access to online research journals."),
    ("faq7", "Transport", "Shuttle Service",
     "The university runs shuttle buses from major city areas to campus every 30 minutes in the morning."),
    ("faq8", "Exams", "Make-up Exams",
     "Students who miss a final exam because of illness can apply for a make-up exam within seven days with a medical certificate."),
    ("faq9", "Career", "Career Services",
     "The career services office organizes job fairs twice a year and helps students prepare resumes and interviews."),
    ("faq10", "IT", "IT Resources",
     "Students receive a university email account and free access to cloud software and Wi-Fi across campus."),
]

# 3. Store each FAQ. Chroma turns the text into a vector automatically.
for faq_id, category, title, text in faqs:
    collection.upsert(
        ids=[faq_id],
        documents=[text],
        metadatas=[{"category": category, "title": title, "source": "University FAQ Handbook"}],
    )
print("Stored", collection.count(), "records in the vector database.\n")


# 4. Search with a question and print the best matches
def search(question, category=None):
    if category:
        results = collection.query(query_texts=[question], n_results=3, where={"category": category})
    else:
        results = collection.query(query_texts=[question], n_results=3)

    print("Question:", question)
    for i in range(len(results["ids"][0])):
        print(f"  {i + 1}. {results['metadatas'][0][i]['title']} (distance: {results['distances'][0][i]:.3f})")
        print(f"     {results['documents'][0][i]}")
    print()


search("How can I get financial help to pay for my studies?")
search("I was sick on the day of my test, what can I do?")
search("Where can I stay if I live in another city?")
search("How do I get help finding a job after graduation?")
search("When can I apply?", category="Admissions")  # similarity search + metadata filter