import { Bell, CheckCircle, AlertCircle, Info, MessageSquare, Award, X, Send } from "lucide-react";
import { useState, useEffect } from "react";
import { getNotifications, markAsRead } from "../services/notificationService";
import { getConversation, sendMessage } from "../services/messageService";

export default function Notifications() {
  const [notifs, setNotifs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedNotification, setSelectedNotification] = useState(null);
  const [replyMessage, setReplyMessage] = useState("");
  const [conversation, setConversation] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [isComposeOpen, setIsComposeOpen] = useState(false);
  const [composeForm, setComposeForm] = useState({ receiverId: "", content: "" });
  const [sendingNew, setSendingNew] = useState(false);

  useEffect(() => {
    fetchNotifs();
  }, []);

  const fetchNotifs = async () => {
    try {
      setLoading(true);
      const data = await getNotifications();
      setNotifs(data.notifications || data || []);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleMarkAsRead = async (id) => {
    try {
      await markAsRead(id);
      setNotifs((prev) => prev.map((n) => n._id === id ? { ...n, isRead: true } : n));
    } catch (error) {
      console.error(error);
    }
  };

  const handleMarkAllAsRead = async () => {
    for (const n of notifs) {
      if (!n.isRead) {
        await handleMarkAsRead(n._id);
      }
    }
  };

  const handleViewMessage = async (notif) => {
    setSelectedNotification(notif);
    setReplyMessage("");
    if (!notif.isRead) {
      handleMarkAsRead(notif._id);
    }
    
    const senderId = notif.sender?._id || notif.sender || notif.senderId;
    if (senderId) {
      setChatLoading(true);
      try {
        const thread = await getConversation(senderId);
        setConversation(thread.messages || thread || []);
      } catch (error) {
        console.error(error);
      } finally {
        setChatLoading(false);
      }
    }
  };

  const handleCloseModal = () => {
    setSelectedNotification(null);
    setReplyMessage("");
    setConversation([]);
  };

  const handleSendReply = async () => {
    if (!replyMessage.trim()) return;
    const senderId = selectedNotification?.sender?._id || selectedNotification?.sender || selectedNotification?.senderId;
    if (!senderId) {
      alert("Cannot reply: Sender ID missing.");
      return;
    }
    try {
      await sendMessage(senderId, replyMessage);
      setReplyMessage("");
      const thread = await getConversation(senderId);
      setConversation(thread.messages || thread || []);
    } catch (error) {
      console.error(error);
    }
  };

  const handleComposeSend = async () => {
    if (!composeForm.receiverId.trim() || !composeForm.content.trim()) return;
    setSendingNew(true);
    try {
      await sendMessage(composeForm.receiverId, composeForm.content);
      alert("Message sent successfully!");
      setIsComposeOpen(false);
      setComposeForm({ receiverId: "", content: "" });
    } catch (error) {
      console.error(error);
      alert("Failed to send message. Please check the Receiver ID.");
    } finally {
      setSendingNew(false);
    }
  };

  const selectedNotif = selectedNotification;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold">Notifications & Messages</h1>
          <p className="mt-2 text-muted-foreground">
            Stay updated with your activities and messages
          </p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={() => setIsComposeOpen(true)} 
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            <MessageSquare className="h-4 w-4" />
            Compose Message
          </button>
          <button 
            onClick={handleMarkAllAsRead} 
            className="rounded-lg border border-primary bg-transparent px-4 py-2 text-primary hover:bg-primary hover:text-primary-foreground transition-colors"
          >
            Mark All as Read
          </button>
        </div>
      </div>

      {/* Notifications List */}
      {loading ? (
        <div className="text-center p-10 text-muted-foreground">Loading notifications...</div>
      ) : notifs.length === 0 ? (
        <div className="text-center p-10 text-muted-foreground">No notifications found.</div>
      ) : (
        <div className="space-y-3">
          {notifs.map((notification, index) => {
            let Icon = Info;
            let color = "text-secondary";
            let bgColor = "bg-secondary/10";
            if (notification.type === "success") { Icon = CheckCircle; color = "text-success"; bgColor = "bg-success/10"; }
            if (notification.type === "event") { Icon = CheckCircle; color = "text-success"; bgColor = "bg-success/10"; }
            if (notification.type === "badge" || notification.type === "achievement") { Icon = Award; color = "text-accent"; bgColor = "bg-accent/10"; }
            if (notification.type === "urgent") { Icon = AlertCircle; color = "text-destructive"; bgColor = "bg-destructive/10"; }
            if (notification.type === "message") { Icon = MessageSquare; color = "text-primary"; bgColor = "bg-primary/10"; }

            return (
              <div
                key={notification._id || index}
                onClick={() => !notification.isRead && handleMarkAsRead(notification._id)}
                className={`rounded-xl border ${notification.isRead ? 'border-border opacity-75' : 'border-primary/50'} bg-card p-4 shadow-sm hover:shadow-lg transition-all cursor-pointer`}
              >
              <div className="flex items-start gap-4">
                <div className={`rounded-lg ${notification.bgColor} p-2.5 flex-shrink-0`}>
                  <Icon className={`h-5 w-5 ${notification.color}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm">{notification.message}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {notification.timestamp}
                  </p>
                </div>
                {notification.type === "message" && (
                  <button 
                    onClick={(e) => { e.stopPropagation(); handleViewMessage(notification); }}
                    className="text-xs text-primary hover:underline flex-shrink-0"
                  >
                    View Message
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
      )}

      {/* Load More */}
      <div className="text-center">
        <button className="rounded-lg border border-border bg-transparent px-6 py-2 text-foreground hover:bg-muted transition-colors">
          Load More Notifications
        </button>
      </div>

      {/* Message Modal */}
      {selectedNotif && selectedNotif.type === "message" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-2xl rounded-xl border border-border bg-card shadow-2xl">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-border p-6">
              <div className="flex items-center gap-3">
                <div className={`rounded-lg ${selectedNotif.bgColor} p-2.5`}>
                  <MessageSquare className={`h-5 w-5 ${selectedNotif.color}`} />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">Message Details</h2>
                  <p className="text-sm text-muted-foreground">{selectedNotif.timestamp}</p>
                </div>
              </div>
              <button
                onClick={handleCloseModal}
                className="rounded-lg p-2 hover:bg-muted transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 space-y-6">
              {/* Message Content */}
              <div className="rounded-lg border border-border bg-background p-4 max-h-[300px] overflow-y-auto flex flex-col gap-3">
                {chatLoading ? (
                  <p className="text-center text-muted-foreground">Loading conversation...</p>
                ) : conversation.length > 0 ? (
                  conversation.map((msg, idx) => (
                    <div key={idx} className={`p-3 rounded-lg text-sm max-w-[80%] ${msg.sender === selectedNotif.recipient ? 'bg-primary text-primary-foreground self-end' : 'bg-muted text-foreground self-start'}`}>
                      {msg.content}
                    </div>
                  ))
                ) : (
                  <div className="space-y-2 text-sm">
                    <p className="text-sm font-semibold mb-2">Message:</p>
                    <p>{selectedNotif.message}</p>
                  </div>
                )}
              </div>

              {/* Reply Section */}
              <div>
                <label htmlFor="reply" className="block text-sm font-semibold mb-2">
                  Your Reply
                </label>
                <textarea
                  id="reply"
                  value={replyMessage}
                  onChange={(e) => setReplyMessage(e.target.value)}
                  placeholder="Type your response here..."
                  rows={4}
                  className="w-full rounded-lg border border-border bg-background px-4 py-3 outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 resize-none"
                />
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-end gap-3">
                <button
                  onClick={handleCloseModal}
                  className="rounded-lg border border-border bg-transparent px-4 py-2 text-foreground hover:bg-muted transition-colors"
                >
                  Close
                </button>
                <button
                  onClick={handleSendReply}
                  disabled={!replyMessage.trim()}
                  className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="h-4 w-4" />
                  Send Reply
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Compose Message Modal */}
      {isComposeOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-lg rounded-xl border border-border bg-card shadow-2xl">
            <div className="flex items-center justify-between border-b border-border p-6">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary/10 p-2.5">
                  <MessageSquare className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">New Message</h2>
                  <p className="text-sm text-muted-foreground">Send a direct message</p>
                </div>
              </div>
              <button
                onClick={() => setIsComposeOpen(false)}
                className="rounded-lg p-2 hover:bg-muted transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-semibold mb-2">Receiver ID</label>
                <input
                  type="text"
                  value={composeForm.receiverId}
                  onChange={(e) => setComposeForm({ ...composeForm, receiverId: e.target.value })}
                  placeholder="Paste User Object ID here..."
                  className="w-full rounded-lg border border-border bg-background px-4 py-2.5 outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-2">Message Content</label>
                <textarea
                  value={composeForm.content}
                  onChange={(e) => setComposeForm({ ...composeForm, content: e.target.value })}
                  placeholder="Type your message here..."
                  rows={5}
                  className="w-full rounded-lg border border-border bg-background px-4 py-3 outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 resize-none"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-4">
                <button
                  onClick={() => setIsComposeOpen(false)}
                  className="rounded-lg border border-border bg-transparent px-4 py-2 text-foreground hover:bg-muted transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleComposeSend}
                  disabled={!composeForm.receiverId.trim() || !composeForm.content.trim() || sendingNew}
                  className="flex items-center gap-2 rounded-lg bg-primary px-6 py-2 text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="h-4 w-4" />
                  {sendingNew ? "Sending..." : "Send Message"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
