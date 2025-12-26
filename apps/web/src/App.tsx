import { useState } from 'react'

import { Button } from './components/ui/button'
import { Input } from './components/ui/input'

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
    <div className='mx-auto flex h-dvh w-full max-w-lg flex-col items-center justify-center gap-y-4'>
      <h1 className='text-4xl font-bold'>Trip Parser</h1>

      <Input type='text' value={text} onChange={e => setText(e.target.value)} />
      <Button onClick={onSubmit}>Submit</Button>

      {response && (
        <div>
          <h2>Response:</h2>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

export default App
