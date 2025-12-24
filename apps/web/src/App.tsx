import { useState } from 'react'

function App() {
  const [text, setText] = useState<string>('')
  const [response, setResponse] = useState(null)

  const onSubmit = async () => {
    try {
      const response = await fetch('/api/trip/parse', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      })
      const data = await response.json()
      setResponse(data)
    } catch (error) {
      console.error('Error during API call:', error)
    }
  }

  return (
    <>
      <input type='text' value={text} onChange={e => setText(e.target.value)} />
      <button onClick={onSubmit}>Submit</button>

      {response && (
        <div>
          <h2>Response:</h2>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </>
  )
}

export default App
