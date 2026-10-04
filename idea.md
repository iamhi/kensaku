I want the project to be avaialbe as a cli cmd like yoakai.
I want this project use minimal dependencies.

This project is a semantic search project.

It should have commands like:

kensaku search "What are good standards for Java?" -- This should trigger kensaku to do semantic search on all workspaces
kensaku search "Whare are good standards for Go?" workspace "development" -- This should trigger kensaku semantic search only on the development workspace

kensaku add workspace cooking "./cooking" -- this adds new workspace with name "cooking" which has directory ./cooking and do semantic search on all of the documents from that directory. After adding a workspace it is expected to have the semantic search indexing.
kensaku remove workspace cooking - this removes a category with name "cooking"

Something to note the contents of the workspace can change. with that we also need to have new indexing so some stype of sha hashing or similar needs to be apploy to the workspace to identify if reindexing is needed. We whould brainstorm this part maybe? or suggest good approach.

kensaku test -- this should test if everything is working well

expect that this project is exposed as skill to llm, claude, copilot, etc...
